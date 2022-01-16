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


class NewToOldSorting:
    """
    This class mimic the old API of spikeextractors with:
      * unique segment
    """
    extractor_name = 'NewToOldSorting'
    is_writable = False

    def __init__(self, sorting):
        assert sorting.get_num_segments() == 1
        self._sorting = sorting
        self._sampling_frequency = sorting.get_sampling_frequency()
        self.is_dumpable = False

        unit_map = {}
        if isinstance(self._sorting.get_unit_ids()[0], (int, np.integer)):
            for u in self._sorting.get_unit_ids():
                unit_map[u] = u
        else:
            for i_u, u in enumerate(self._sorting.get_unit_ids()):
                unit_map[i_u] = u
        self._unit_map = unit_map

    def get_unit_ids(self):
        """This function returns a list of ids (ints) for each unit in the sorsted result.

        Returns
        -------
        unit_ids: array_like
            A list of the unit ids in the sorted result (ints).
        """
        return list(self._unit_map.keys())

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        """This function extracts spike frames from the specified unit.
        It will return spike frames from within three ranges:

            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_unit_spike_frame - 1]

        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Spike frames are returned in the form of an
        array_like of spike frames. In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the recording
        start_frame: int
            The frame above which a spike frame is returned  (inclusive)
        end_frame: int
            The frame below which a spike frame is returned  (exclusive)

        Returns
        -------
        spike_train: numpy.ndarray
            An 1D array containing all the frames for each spike in the
            specified unit given the range of start and end frames
        """
        return self._sorting.get_unit_spike_train(unit_id=self._unit_map[unit_id], segment_index=0,
                                                  start_frame=start_frame, end_frame=end_frame)

    def get_units_spike_train(self, unit_ids=None, start_frame=None, end_frame=None):
        """This function extracts spike frames from the specified units.

        Parameters
        ----------
        unit_ids: array_like
            The unit ids from which to return spike trains. If None, all unit
            spike trains will be returned
        start_frame: int
            The frame above which a spike frame is returned  (inclusive)
        end_frame: int
            The frame below which a spike frame is returned  (exclusive)

        Returns
        -------
        spike_train: numpy.ndarray
            An 2D array containing all the frames for each spike in the
            specified units given the range of start and end frames
        """
        if unit_ids is None:
            unit_ids = self.get_unit_ids()
        spike_trains = [self.get_unit_spike_train(uid, start_frame, end_frame) for uid in unit_ids]
        return spike_trains

    def get_sampling_frequency(self):
        """
        It returns the sampling frequency.

        Returns
        -------
        sampling_frequency: float
            The sampling frequency
        """
        return self._sampling_frequency

    def set_sampling_frequency(self, sampling_frequency):
        """
        It sets the sorting extractor sampling frequency.

        Parameters
        ----------
        sampling_frequency: float
            The sampling frequency
        """
        self._sampling_frequency = sampling_frequency

    def set_times(self, times):
        """This function sets the sorting times to convert spike trains to seconds

        Parameters
        ----------
        times: array-like
            The times in seconds for each frame
        """
        max_frames = np.array([np.max(self.get_unit_spike_train(u)) for u in self.get_unit_ids()])
        assert np.all(max_frames < len(times)), "The length of 'times' should be greater than the maximum " \
                                                "spike frame index"
        self._times = times.astype('float64')

    def frame_to_time(self, frames):
        """This function converts user-inputed frame indexes to times with units of seconds.

        Parameters
        ----------
        frames: float or array-like
            The frame or frames to be converted to times

        Returns
        -------
        times: float or array-like
            The corresponding times in seconds
        """
        # Default implementation
        if self._times is None:
            return np.round(frames / self.get_sampling_frequency(), 6)
        else:
            return self._times[frames]

    def time_to_frame(self, times):
        """This function converts a user-inputted times (in seconds) to a frame indexes.

        Parameters
        ----------
        times: float or array-like
            The times (in seconds) to be converted to frame indexes

        Returns
        -------
        frames: float or array-like
            The corresponding frame indexes
        """
        # Default implementation
        if self._times is None:
            return np.round(times * self.get_sampling_frequency()).astype('int64')
        else:
            return np.searchsorted(self._times, times).astype('int64')


def create_extractor_from_new_sorting(new_sorting):
    old_sorting = NewToOldSorting(new_sorting)
    return old_sorting


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
        copy_properties(oldapi_extractor=oldapi_recording_extractor, new_extractor=self,
                        old_to_new_property_map=_old_to_new_property_map)

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
                                                           end_frame=end_frame,
                                                           return_scaled=False).T
        
def create_recording_from_old_extractor(oldapi_recording_extractor)->OldToNewRecording:
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
        copy_properties(oldapi_extractor=oldapi_sorting_extractor, new_extractor=self)

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


def copy_properties(oldapi_extractor, new_extractor, old_to_new_property_map={}):
    # add old properties
    properties = dict()
    if hasattr(oldapi_extractor, "get_channel_ids"):
        get_ids = oldapi_extractor.get_channel_ids
        get_property = oldapi_extractor.get_channel_property
        get_property_names = oldapi_extractor.get_channel_property_names
    else:
        get_ids = oldapi_extractor.get_unit_ids
        get_property = oldapi_extractor.get_unit_property
        get_property_names = oldapi_extractor.get_unit_property_names

    for id in get_ids():
        properties_for_channel = get_property_names(id)
        for prop in properties_for_channel:
            prop_value = get_property(id, prop)
            skip_if_value = None
            if prop in old_to_new_property_map:
                prop_name = old_to_new_property_map[prop]["name"]
                skip_if_value = old_to_new_property_map[prop]["skip_if_value"]
            else:
                prop_name = prop

            if skip_if_value is not None:
                if prop_value == skip_if_value:
                    continue

            if prop_name not in properties:
                properties[prop_name] = dict()
                properties[prop_name]["ids"] = []
                properties[prop_name]["values"] = []

            properties[prop_name]["ids"].append(id)
            properties[prop_name]["values"].append(prop_value)

    for property_name, prop_dict in properties.items():
        property_ids = prop_dict["ids"]
        property_values = prop_dict["values"]
        new_extractor.set_property(key=property_name,
                                   values=property_values, ids=property_ids)
