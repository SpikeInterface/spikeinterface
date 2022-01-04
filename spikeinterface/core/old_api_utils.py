from typing import Union

import numpy as np
from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment)


class NewToOldRecording:
    """
    This class mimic the old API of spikeextractors with:
      * reversed shape (channels, samples):
      * unique segment
    This is internally used for:
      * Montainsort4
      * Herdinspikes
    Because these sorters are based on the old API
    """

    def __init__(self, recording):
        assert recording.get_num_segments() == 1
        self._recording = recording

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame, end_frame=end_frame,
                                            segment_index=0)
        return traces.T

    def get_num_frames(self):
        return self._recording.get_num_frames(segment_index=0)

    def get_num_channels(self):
        return self._recording.get_num_channels()

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_channel_property(self, channel_id, property):
        rec = self._recording
        values = rec.get_property(property)
        ind = rec.ids_to_indices([channel_id])
        v = values[ind[0]]
        return v


def create_extractor_from_new_recording(new_recording):
    old_recording = NewToOldRecording(new_recording)
    return old_recording


_old_to_new_property_map = {'gain': {'name': 'gain_to_uV', 'skip_if_value': 1},
                            'offset': {'name': 'offset_to_uV', 'skip_if_value': 0}}


class OldToNewRecording(BaseRecording):
    """Wrapper class to convert old RecordingExtractor to a
    new Recording in spikeinterface > v0.90

    Parameters
    ----------
    oldapi_recording_extractor : se.RecordingExtractor
        recording extractor from spikeinterface < v0.90
    """

    def __init__(self, oldapi_recording_extractor):
        BaseRecording.__init__(self, oldapi_recording_extractor.get_sampling_frequency(),
                               oldapi_recording_extractor.get_channel_ids(),
                               oldapi_recording_extractor.get_dtype())

        # set is_dumpable to False to use dumping mechanism of old extractor
        self.is_dumpable = False
        self.annotate(is_filtered=oldapi_recording_extractor.is_filtered)

        # add old recording as a recording segment
        recording_segment = OldToNewRecordingSegment(oldapi_recording_extractor)
        self.add_recording_segment(recording_segment)
        self.set_channel_locations(oldapi_recording_extractor.get_channel_locations())

        # add old properties
        properties = dict()
        for ch in oldapi_recording_extractor.get_channel_ids():
            properties_for_channel = oldapi_recording_extractor.get_channel_property_names(ch)
            for prop in properties_for_channel:
                prop_value = oldapi_recording_extractor.get_channel_property(ch, prop)
                skip_if_value = None
                if prop in _old_to_new_property_map:
                    prop_name = _old_to_new_property_map[prop]["name"]
                    skip_if_value = _old_to_new_property_map[prop]["skip_if_value"]
                else:
                    prop_name = prop

                if prop_name not in properties:
                    properties[prop_name] = dict()
                    properties[prop_name]["ids"] = []
                    properties[prop_name]["values"] = []

                if skip_if_value is not None:
                    if prop_value == skip_if_value:
                        del properties[prop_name]
                        break
                properties[prop_name]["ids"].append(ch)
                properties[prop_name]["values"].append(prop_value)

        for property_name, prop_dict in properties.items():
            property_ids = prop_dict["ids"]
            property_values = prop_dict["values"]
            self.set_property(key=property_name, values=property_values, ids=property_ids)

        self._kwargs = {'oldapi_recording_extractor': oldapi_recording_extractor}


class OldToNewRecordingSegment(BaseRecordingSegment):
    """Wrapper class to convert old RecordingExtractor to a
    RecordingSegment in spikeinterface > v0.90

    Parameters
    ----------
    oldapi_recording_extractor : se.RecordingExtractor
        recording extractor from spikeinterface < v0.90
    """
    def __init__(self, oldapi_recording_extractor):
        BaseRecordingSegment.__init__(self, sampling_frequency=oldapi_recording_extractor.get_sampling_frequency(),
                                      t_start=None, time_vector=None)
        self._oldapi_recording_extractor = oldapi_recording_extractor
        self._channel_ids = np.array(oldapi_recording_extractor.get_channel_ids())

        self._kwargs = {'oldapi_recording_extractor': oldapi_recording_extractor}

    def get_num_samples(self):
        return self._oldapi_recording_extractor.get_num_frames()

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_ids = self._channel_ids
        else:
            channel_ids = self._channel_ids[channel_indices]
        return self._oldapi_recording_extractor.get_traces(channel_ids=channel_ids,
                                                           start_frame=start_frame,
                                                           end_frame=end_frame).T


def create_recording_from_old_extractor(oldapi_recording_extractor) -> OldToNewRecording:
    new_recording = OldToNewRecording(oldapi_recording_extractor)
    return new_recording


class OldToNewSorting(BaseSorting):
    """Wrapper class to convert old SortingExtractor to a
    new Sorting in spikeinterface > v0.90

    Parameters
    ----------
    oldapi_sorting_extractor : se.SortingExtractor
        sorting extractor from spikeinterface < v0.90
    """

    def __init__(self, oldapi_sorting_extractor):
        BaseSorting.__init__(self, sampling_frequency=oldapi_sorting_extractor.get_sampling_frequency(),
                             unit_ids=oldapi_sorting_extractor.get_unit_ids())

        sorting_segment = OldToNewSortingSegment(oldapi_sorting_extractor)
        self.add_sorting_segment(sorting_segment)

        self.is_dumpable = False

        # add old properties
        properties = dict()
        for unit in oldapi_sorting_extractor.get_unit_ids():
            properties_for_unit = oldapi_sorting_extractor.get_unit_property_names(unit)
            for prop in properties_for_unit:
                if prop not in properties:
                    properties[prop] = dict()
                    properties[prop]["ids"] = []
                    properties[prop]["values"] = []
                properties[prop]["ids"].append(unit)
                properties[prop]["values"].append(oldapi_sorting_extractor.get_unit_property(unit, prop))

        for property_name, prop_dict in properties.items():
            property_ids = prop_dict["ids"]
            property_values = prop_dict["values"]
            self.set_property(key=property_name, values=property_values, ids=property_ids)

        self._kwargs = {'oldapi_sorting_extractor': oldapi_sorting_extractor}


class OldToNewSortingSegment(BaseSortingSegment):
    """Wrapper class to convert old SortingExtractor to a
    SortingSegment in spikeinterface > v0.90

    Parameters
    ----------
    oldapi_sorting_extractor : se.SortingExtractor
        sorting extractor from spikeinterface < v0.90
    """
    def __init__(self, oldapi_sorting_extractor):
        BaseSortingSegment.__init__(self)
        self._oldapi_sorting_extractor = oldapi_sorting_extractor

        self._kwargs = {'oldapi_sorting_extractor': oldapi_sorting_extractor}

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:

        return self._oldapi_sorting_extractor.get_unit_spike_train(unit_id=unit_id,
                                                                   start_frame=start_frame,
                                                                   end_frame=end_frame)


def create_sorting_from_old_extractor(oldapi_sorting_extractor) -> OldToNewSorting:
    new_sorting = OldToNewSorting(oldapi_sorting_extractor)
    return new_sorting
