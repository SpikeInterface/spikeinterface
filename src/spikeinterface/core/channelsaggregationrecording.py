from __future__ import annotations

import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment


class ChannelsAggregationRecording(BaseRecording):
    """
    Class that handles aggregating channels from different recordings, e.g. from different channel groups.

    Do not use this class directly but use `si.aggregate_channels(...)`

    """

    def __init__(self, recording_list, renamed_channel_ids=None):

        self._recordings = recording_list

        self._perform_consistency_checks()
        sampling_frequency = recording_list[0].get_sampling_frequency()
        dtype = recording_list[0].get_dtype()
        num_segments = recording_list[0].get_num_segments()

        # Generate a default list of channel ids that are unique and consecutive numbers as strings.
        num_all_channels = sum(rec.get_num_channels() for rec in recording_list)
        if renamed_channel_ids is not None:
            assert (
                len(np.unique(renamed_channel_ids)) == num_all_channels
            ), "'renamed_channel_ids' doesn't have the right size or has duplicates!"
            channel_ids = list(renamed_channel_ids)
        else:
            # Collect channel IDs from all recordings
            all_channels_have_same_type = np.unique([rec.channel_ids.dtype for rec in recording_list]).size == 1
            all_channel_ids_are_unique = False
            if all_channels_have_same_type:
                combined_ids = np.concatenate([rec.channel_ids for rec in recording_list])
                all_channel_ids_are_unique = np.unique(combined_ids).size == num_all_channels

            if all_channels_have_same_type and all_channel_ids_are_unique:
                channel_ids = combined_ids
            else:
                # If IDs are not unique or not of the same type, use default as stringify IDs
                default_channel_ids = [str(i) for i in range(num_all_channels)]
                channel_ids = default_channel_ids

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
                            property_dict[prop_name] = np.concatenate(
                                (property_dict[prop_name], rec.get_property(prop_name))
                            )
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
        if "location" in self.get_property_keys():
            location_tuple = [tuple(loc) for loc in self.get_property("location")]
            assert len(set(location_tuple)) == self.get_num_channels(), (
                "Locations are not unique! " "Cannot aggregate recordings!"
            )

        # finally add segments, we need a channel mapping
        ch_id = 0
        channel_map = {}
        for r_i, recording in enumerate(recording_list):
            single_channel_ids = recording.get_channel_ids()
            single_channel_indices = recording.ids_to_indices(single_channel_ids)
            for chan_id, chan_idx in zip(single_channel_ids, single_channel_indices):
                channel_map[ch_id] = {"recording_id": r_i, "channel_index": chan_idx}
                ch_id += 1

        for i_seg in range(num_segments):
            parent_segments = [rec._recording_segments[i_seg] for rec in recording_list]
            sub_segment = ChannelsAggregationRecordingSegment(channel_map, parent_segments)
            self.add_recording_segment(sub_segment)

        self._kwargs = {"recording_list": recording_list, "renamed_channel_ids": renamed_channel_ids}

    @property
    def recordings(self):
        return self._recordings

    def _perform_consistency_checks(self):

        # Check for consistent sampling frequency across recordings
        sampling_frequencies = [rec.get_sampling_frequency() for rec in self.recordings]
        sampling_frequency = sampling_frequencies[0]
        consistent_sampling_frequency = all(sampling_frequency == sf for sf in sampling_frequencies)
        if not consistent_sampling_frequency:
            raise ValueError(f"Inconsistent sampling frequency among recordings: {sampling_frequencies}")

        # Check for consistent number of segments across recordings
        num_segments_list = [rec.get_num_segments() for rec in self.recordings]
        num_segments = num_segments_list[0]
        consistent_num_segments = all(num_segments == ns for ns in num_segments_list)
        if not consistent_num_segments:
            raise ValueError(f"Inconsistent number of segments among recordings: {num_segments_list}")

        # Check for consistent data type across recordings
        data_types = [rec.get_dtype() for rec in self.recordings]
        dtype = data_types[0]
        consistent_dtype = all(dtype == dt for dt in data_types)
        if not consistent_dtype:
            raise ValueError(f"Inconsistent data type among recordings: {data_types}")

        # Check for consistent number of samples across recordings for each segment
        for segment_index in range(num_segments):
            num_samples_list = [rec.get_num_samples(segment_index=segment_index) for rec in self.recordings]
            num_samples = num_samples_list[0]
            consistent_num_samples = all(num_samples == ns for ns in num_samples_list)
            if not consistent_num_samples:
                raise ValueError(
                    f"Inconsistent number of samples in segment {segment_index} among recordings: {num_samples_list}"
                )


class ChannelsAggregationRecordingSegment(BaseRecordingSegment):
    """
    Class to return a aggregated segment traces.
    """

    def __init__(self, channel_map, parent_segments):
        parent_segment0 = parent_segments[0]
        times_kargs0 = parent_segment0.get_times_kwargs()
        if times_kargs0["time_vector"] is None:
            for ps in parent_segments:
                assert ps.get_times_kwargs()["time_vector"] is None, "All segments should not have times set"
        else:
            for ps in parent_segments:
                assert ps.get_times_kwargs()["t_start"] == times_kargs0["t_start"], (
                    "All segments should have the same " "t_start"
                )

        BaseRecordingSegment.__init__(self, **times_kargs0)
        self._channel_map = channel_map
        self._parent_segments = parent_segments

    def get_num_samples(self) -> int:
        # num samples are all the same
        return self._parent_segments[0].get_num_samples()

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | slice | None = None,
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
            recording_id_channels_map = {}
            for channel_idx in channel_indices:
                recording_id = self._channel_map[channel_idx]["recording_id"]
                channel_index_recording = self._channel_map[channel_idx]["channel_index"]
                if recording_id not in recording_id_channels_map:
                    recording_id_channels_map[recording_id] = []
                recording_id_channels_map[recording_id].append(channel_index_recording)
            for recording_id, channel_indices_recording in recording_id_channels_map.items():
                segment = self._parent_segments[recording_id]
                traces_recording = segment.get_traces(
                    channel_indices=channel_indices_recording, start_frame=start_frame, end_frame=end_frame
                )
                traces.append(traces_recording)
        else:
            for segment in self._parent_segments:
                traces_all_recording = segment.get_traces(
                    channel_indices=channel_indices, start_frame=start_frame, end_frame=end_frame
                )
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
        If given, channel ids are renamed as provided.

    Returns
    -------
    aggregate_recording: ChannelsAggregationRecording
        The aggregated recording object
    """
    return ChannelsAggregationRecording(recording_list, renamed_channel_ids)
