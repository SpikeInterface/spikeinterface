from __future__ import annotations

from typing import Union

import numpy as np
import warnings
from spikeinterface.core import BaseRecording, BaseSorting, BaseRecordingSegment, BaseSortingSegment


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
        traces = self._recording.get_traces(
            channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame, segment_index=0
        )
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

    def __init__(self, sorting):
        assert sorting.get_num_segments() == 1
        self._sorting = sorting
        self._sampling_frequency = sorting.get_sampling_frequency()

        unit_map = {}
        if np.all([isinstance(unit_id, int)] for unit_id in self._sorting.get_unit_ids()):
            for u in self._sorting.get_unit_ids():
                unit_map[u] = u
        else:
            print(
                "Some unit IDs are not int but all unit IDs must be int in the old API SortingExtractor. Converting unit IDs to index..."
            )
            for i_u, u in enumerate(self._sorting.get_unit_ids()):
                unit_map[i_u] = u
        self._unit_map = unit_map

        self._kwargs = dict(sorting=sorting)

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
        return self._sorting.get_unit_spike_train(
            unit_id=self._unit_map[unit_id], segment_index=0, start_frame=start_frame, end_frame=end_frame
        )

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


def create_extractor_from_new_sorting(new_sorting):
    old_sorting = NewToOldSorting(new_sorting)
    return old_sorting


class OldToNewRecording(BaseRecording):
    """Wrapper class to convert old RecordingExtractor to a
    new Recording in spikeinterface > v0.90

    Parameters
    ----------
    oldapi_recording_extractor : se.RecordingExtractor
        recording extractor from spikeinterface < v0.90
    """

    def __init__(self, oldapi_recording_extractor):
        BaseRecording.__init__(
            self,
            sampling_frequency=float(oldapi_recording_extractor.get_sampling_frequency()),
            channel_ids=oldapi_recording_extractor.get_channel_ids(),
            dtype=oldapi_recording_extractor.get_dtype(return_scaled=False),
        )

        # set to False to use dumping mechanism of old extractor
        self._serializability["memory"] = False
        self._serializability["json"] = False
        self._serializability["pickle"] = False

        self.annotate(is_filtered=oldapi_recording_extractor.is_filtered)

        # add old recording as a recording segment
        recording_segment = OldToNewRecordingSegment(oldapi_recording_extractor)
        self.add_recording_segment(recording_segment)
        self.set_channel_locations(oldapi_recording_extractor.get_channel_locations())

        # add old properties
        copy_properties(
            oldapi_extractor=oldapi_recording_extractor, new_extractor=self, skip_properties=["gain", "offset"]
        )
        # set correct gains and offsets
        gains, offsets = find_old_gains_offsets_recursively(oldapi_recording_extractor.dump_to_dict())
        if gains is not None:
            if np.any(gains != 1):
                self.set_channel_gains(gains)
        if offsets is not None:
            if np.any(offsets != 0):
                self.set_channel_offsets(offsets)

        self._kwargs = {"oldapi_recording_extractor": oldapi_recording_extractor}


class OldToNewRecordingSegment(BaseRecordingSegment):
    """Wrapper class to convert old RecordingExtractor to a
    RecordingSegment in spikeinterface > v0.90

    Parameters
    ----------
    oldapi_recording_extractor : se.RecordingExtractor
        recording extractor from spikeinterface < v0.90
    """

    def __init__(self, oldapi_recording_extractor):
        BaseRecordingSegment.__init__(
            self,
            sampling_frequency=float(oldapi_recording_extractor.get_sampling_frequency()),
            t_start=None,
            time_vector=None,
        )
        self._oldapi_recording_extractor = oldapi_recording_extractor
        self._channel_ids = np.array(oldapi_recording_extractor.get_channel_ids())

        self._kwargs = {"oldapi_recording_extractor": oldapi_recording_extractor}

    def get_num_samples(self):
        return self._oldapi_recording_extractor.get_num_frames()

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_ids = self._channel_ids
        else:
            channel_ids = self._channel_ids[channel_indices]
        return self._oldapi_recording_extractor.get_traces(
            channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame, return_scaled=False
        ).T


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
        BaseSorting.__init__(
            self,
            sampling_frequency=float(oldapi_sorting_extractor.get_sampling_frequency()),
            unit_ids=oldapi_sorting_extractor.get_unit_ids(),
        )

        sorting_segment = OldToNewSortingSegment(oldapi_sorting_extractor)
        self.add_sorting_segment(sorting_segment)

        self._serializability["memory"] = False
        self._serializability["json"] = False
        self._serializability["pickle"] = False

        # add old properties
        copy_properties(oldapi_extractor=oldapi_sorting_extractor, new_extractor=self)

        self._kwargs = {"oldapi_sorting_extractor": oldapi_sorting_extractor}


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

        self._kwargs = {"oldapi_sorting_extractor": oldapi_sorting_extractor}

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
    ) -> np.ndarray:
        return self._oldapi_sorting_extractor.get_unit_spike_train(
            unit_id=unit_id, start_frame=start_frame, end_frame=end_frame
        )


def create_sorting_from_old_extractor(oldapi_sorting_extractor) -> OldToNewSorting:
    new_sorting = OldToNewSorting(oldapi_sorting_extractor)
    return new_sorting


def copy_properties(oldapi_extractor, new_extractor, skip_properties=None):
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

    if skip_properties is None:
        skip_properties = []

    for id in get_ids():
        properties_for_channel = get_property_names(id)
        for prop_name in properties_for_channel:
            prop_value = get_property(id, prop_name)

            if prop_name in skip_properties:
                continue

            if prop_name not in properties:
                properties[prop_name] = dict()
                properties[prop_name]["ids"] = []
                properties[prop_name]["values"] = []

            properties[prop_name]["ids"].append(id)
            properties[prop_name]["values"].append(prop_value)

    for property_name, prop_dict in properties.items():
        property_ids = np.array(prop_dict["ids"])
        property_values = np.array(prop_dict["values"])
        missing_value = None

        # For back-compatibility, incomplete int/uint properties are upcast to float
        # and missing_value is set to np.nan
        if len(property_ids) < len(get_ids()):
            if property_values.dtype.kind in ("u", "i"):
                property_values = property_values.astype("float")
                missing_value = np.nan
        try:
            new_extractor.set_property(
                key=property_name, values=property_values, ids=property_ids, missing_value=missing_value
            )
        except Exception as e:
            warnings.warn(f"Property {property_name} cannot be ported to new API due to missing values.")


def find_old_gains_offsets_recursively(oldapi_extractor_dict):
    kwargs = oldapi_extractor_dict["kwargs"]
    if np.any([isinstance(v, dict) and "dumpable" in v.keys() for (k, v) in kwargs.items()]):
        # check nested
        for k, v in oldapi_extractor_dict["kwargs"].items():
            if isinstance(v, dict) and "dumpable" in v:
                return find_old_gains_offsets_recursively(v)
    else:
        gains = oldapi_extractor_dict["key_properties"]["gain"]
        offsets = oldapi_extractor_dict["key_properties"]["offset"]

        return gains, offsets
