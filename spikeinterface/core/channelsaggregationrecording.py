from typing import List, Union

import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment


class ChannelsAggregationRecording(BaseRecording):
    """
    Class that handles aggregating channels from different recordings, e.g. from different channel groups.

    Do not use this class directly but use `si.aggregate_channels(...)`

    """
    def __init__(self, recording_list, renamed_channel_ids=None):
        channel_map = {}

        num_all_channels = sum([rec.get_num_channels() for rec in recording_list])
        if renamed_channel_ids is not None:
            assert len(np.unique(renamed_channel_ids)) == num_all_channels, "'renamed_channel_ids' doesn't have the " \
                                                                            "right size or has duplicates!"
            channel_ids = list(renamed_channel_ids)
        else:
            channel_ids = list(np.arange(num_all_channels))

        # channel map maps channel indices that are used to get traces
        ch_id = 0
        for r_i, recording in enumerate(recording_list):
            single_channel_ids = recording.get_channel_ids()
            single_channel_indices = recording.ids_to_indices(single_channel_ids)
            for (chan_id, chan_idx) in zip(single_channel_ids, single_channel_indices):
                channel_map[ch_id] = {'recording_id': r_i, 'channel_index': chan_idx}
                ch_id += 1

        sampling_frequency = recording_list[0].get_sampling_frequency()
        num_segments = recording_list[0].get_num_segments()
        dtype = recording_list[0].get_dtype()

        ok1 = all(sampling_frequency == rec.get_sampling_frequency() for rec in recording_list)
        ok2 = all(num_segments == rec.get_num_segments() for rec in recording_list)
        ok3 = all(dtype == rec.get_dtype() for rec in recording_list)
        ok4 = True
        for i_seg in range(num_segments):
            num_samples = recording_list[0].get_num_samples(i_seg)
            ok4 = all(num_samples == rec.get_num_samples(i_seg) for rec in recording_list)
            if not ok4:
                break

        if not (ok1 and ok2 and ok3 and ok4):
            raise ValueError("Sortings don't have the same sampling_frequency/num_segments/dtype/num samples")

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        property_keys = recording_list[0].get_property_keys()
        property_dict = {}
        for prop_name in property_keys:
            if all([prop_name in rec.get_property_keys() for rec in recording_list]):
                for i_r, rec in enumerate(recording_list):
                    prop_value = rec.get_property(prop_name)
                    if i_r == 0:
                        property_dict[prop_name] = prop_value
                    else:
                        try:
                            property_dict[prop_name] = np.concatenate((property_dict[prop_name],
                                                                       rec.get_property(prop_name)))
                        except Exception as e:
                            print(f"Skipping property '{prop_name}' for shape inconsistency")
                            del property_dict[prop_name]
                            break

        for prop_name, prop_values in property_dict.items():
            if prop_name == "contact_vector":
                # remap device channel indices correctly
                prop_values["device_channel_indices"] = np.arange(self.get_num_channels())
            self.set_property(key=prop_name, values=prop_values)

        # if locations are present, check that they are all different!
        if 'location' in self.get_property_keys():
            location_tuple = [tuple(loc) for loc in self.get_property('location')]
            assert len(set(location_tuple)) == self.get_num_channels(), "Locations are not unique! " \
                                                                        "Cannot aggregate recordings!"

        # finally add segments
        for i_seg in range(num_segments):
            parent_segments = [rec._recording_segments[i_seg] for rec in recording_list]
            sub_segment = ChannelsAggregationRecordingSegment(channel_map, parent_segments)
            self.add_recording_segment(sub_segment)

        self._recordings = recording_list
        self._kwargs = {'recording_list': [rec.to_dict() for rec in recording_list],
                        'renamed_channel_ids': renamed_channel_ids}


class ChannelsAggregationRecordingSegment(BaseRecordingSegment):
    """
    Class to return a aggregated segment traces.
    """

    def __init__(self, channel_map, parent_segments):
        parent_segment0 = parent_segments[0]
        times_kargs0 = parent_segment0.get_times_kwargs()
        if times_kargs0['time_vector'] is None:
            for ps in parent_segments:
                assert ps.get_times_kwargs()['time_vector'] is None, "All segment should not have times set"
        else:
            for ps in parent_segments:
                assert ps.get_times_kwargs()['t_start'] == times_kargs0['t_start'], "All segment should have the same "\
                                                                                    "t_start"
            
        BaseRecordingSegment.__init__(self, **times_kargs0)
        self._channel_map = channel_map
        self._parent_segments = parent_segments

    def get_num_samples(self) -> int:
        # num samples are all the same
        return self._parent_segments[0].get_num_samples()

    def get_traces(self,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None,
                   ) -> np.ndarray:

        return_all_channels = False
        if channel_indices is None:
            return_all_channels = True
        elif isinstance(channel_indices, slice):
            if channel_indices == slice(None, None, None):
                return_all_channels = True

        traces = []
        if not return_all_channels:
            if isinstance(channel_indices, slice):
                # in case channel_indices is slice, it has step 1
                step = channel_indices.step if channel_indices.step is not None else 1
                channel_indices = list(range(channel_indices.start, channel_indices.stop, step))
            for channel_idx in channel_indices:
                segment = self._parent_segments[self._channel_map[channel_idx]['recording_id']]
                channel_index_recording = self._channel_map[channel_idx]['channel_index']
                traces_recording = segment.get_traces(channel_indices=[channel_index_recording],
                                                      start_frame=start_frame,
                                                      end_frame=end_frame)
                traces.append(traces_recording)
        else:
            for segment in self._parent_segments:
                traces_all_recording = segment.get_traces(channel_indices=channel_indices,
                                                          start_frame=start_frame,
                                                          end_frame=end_frame)
                traces.append(traces_all_recording)
        return np.concatenate(traces, axis=1)


def aggregate_channels(recording_list, renamed_channel_ids=None):
    """
    Aggregates channels of multiple recording into a single recording object

    Parameters
    ----------
    recording_list: list
        List of BaseRecording objects to aggregate
    renamed_channel_ids: array-like
        If given, channel ids are renamed as provided. If None, unit ids are sequential integers.

    Returns
    -------
    aggregate_recording: UnitsAggregationSorting
        The aggregated sorting object
    """
    return ChannelsAggregationRecording(recording_list, renamed_channel_ids)
