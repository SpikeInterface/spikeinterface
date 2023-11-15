from __future__ import annotations

import warnings
from typing import List, Optional, Union

import numpy as np

from .base import BaseExtractor, BaseSegment
from .waveform_tools import has_exceeding_spikes


minimum_spike_dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]


class BaseSorting(BaseExtractor):
    """
    Abstract class representing several segment several units and relative spiketrains.
    """

    def __init__(self, sampling_frequency: float, unit_ids: List):
        BaseExtractor.__init__(self, unit_ids)
        self._sampling_frequency = sampling_frequency
        self._sorting_segments: List[BaseSortingSegment] = []
        # this weak link is to handle times from a recording object
        self._recording = None
        self._sorting_info = None

        # caching
        self._cached_spike_vector = None
        self._cached_spike_trains = {}

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nunits = self.get_num_units()
        sf_khz = self.get_sampling_frequency() / 1000.0
        txt = f"{clsname}: {nunits} units - {nseg} segments - {sf_khz:0.1f}kHz"
        if "file_path" in self._kwargs:
            txt += "\n  file_path: {}".format(self._kwargs["file_path"])
        return txt

    @property
    def unit_ids(self):
        return self._main_ids

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def get_unit_ids(self) -> List:
        return self._main_ids

    def get_num_units(self) -> int:
        return len(self.get_unit_ids())

    def add_sorting_segment(self, sorting_segment):
        self._sorting_segments.append(sorting_segment)
        sorting_segment.set_parent_extractor(self)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_num_segments(self):
        return len(self._sorting_segments)

    def get_num_samples(self, segment_index=None):
        """Returns the number of samples of the associated recording for a segment.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index to retrieve the number of samples for.
            For multi-segment objects, it is required

        Returns
        -------
        int
            The number of samples
        """
        assert (
            self.has_recording()
        ), "This methods requires an associated recording. Call self.register_recording() first."
        return self._recording.get_num_samples(segment_index=segment_index)

    def get_total_samples(self):
        """Returns the total number of samples of the associated recording.

        Returns
        -------
        int
            The total number of samples
        """
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_samples(segment_index)
        return s

    def get_total_duration(self):
        """Returns the total duration in s of the associated recording.

        Returns
        -------
        float
            The duration in seconds
        """
        assert (
            self.has_recording()
        ), "This methods requires an associated recording. Call self.register_recording() first."
        return self._recording.get_total_duration()

    def get_unit_spike_train(
        self,
        unit_id,
        segment_index: Union[int, None] = None,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        return_times: bool = False,
        use_cache: bool = True,
    ):
        segment_index = self._check_segment_index(segment_index)
        if use_cache:
            if segment_index not in self._cached_spike_trains:
                self._cached_spike_trains[segment_index] = {}
            if unit_id not in self._cached_spike_trains[segment_index]:
                segment = self._sorting_segments[segment_index]
                spike_frames = segment.get_unit_spike_train(unit_id=unit_id, start_frame=None, end_frame=None).astype(
                    "int64"
                )
                self._cached_spike_trains[segment_index][unit_id] = spike_frames
            else:
                spike_frames = self._cached_spike_trains[segment_index][unit_id]
            if start_frame is not None:
                spike_frames = spike_frames[spike_frames >= start_frame]
            if end_frame is not None:
                spike_frames = spike_frames[spike_frames < end_frame]
        else:
            segment = self._sorting_segments[segment_index]
            spike_frames = segment.get_unit_spike_train(
                unit_id=unit_id, start_frame=start_frame, end_frame=end_frame
            ).astype("int64")

        if return_times:
            if self.has_recording():
                times = self.get_times(segment_index=segment_index)
                return times[spike_frames]
            else:
                segment = self._sorting_segments[segment_index]
                t_start = segment._t_start if segment._t_start is not None else 0
                spike_times = spike_frames / self.get_sampling_frequency()
                return t_start + spike_times
        else:
            return spike_frames

    def register_recording(self, recording, check_spike_frames=True):
        """Register a recording to the sorting.

        Parameters
        ----------
        recording : BaseRecording
            Recording with the same number of segments as current sorting.
            Assigned to self._recording.
        check_spike_frames : bool, default: True
            If True, assert for each segment that all spikes are within the recording's range.
        """
        assert np.isclose(
            self.get_sampling_frequency(), recording.get_sampling_frequency(), atol=0.1
        ), "The recording has a different sampling frequency than the sorting!"
        assert (
            self.get_num_segments() == recording.get_num_segments()
        ), "The recording has a different number of segments than the sorting!"
        if check_spike_frames:
            if has_exceeding_spikes(recording, self):
                warnings.warn(
                    "Some spikes exceed the recording's duration! "
                    "Removing these excess spikes with `spikeinterface.curation.remove_excess_spikes()` "
                    "Might be necessary for further postprocessing."
                )
        self._recording = recording

    @property
    def sorting_info(self):
        if "__sorting_info__" in self.get_annotation_keys():
            return self.get_annotation("__sorting_info__")
        else:
            return None

    def set_sorting_info(self, recording_dict, params_dict, log_dict):
        sorting_info = dict(recording=recording_dict, params=params_dict, log=log_dict)
        self.annotate(__sorting_info__=sorting_info)

    def has_recording(self):
        return self._recording is not None

    def has_time_vector(self, segment_index=None):
        """
        Check if the segment of the registered recording has a time vector.
        """
        segment_index = self._check_segment_index(segment_index)
        if self.has_recording():
            return self._recording.has_time_vector(segment_index=segment_index)
        else:
            return False

    def get_times(self, segment_index=None):
        """
        Get time vector for a registered recording segment.

        If a recording is registered:
            * if the segment has a time_vector, then it is returned
            * if not, a time_vector is constructed on the fly with sampling frequency

        If there is no registered recording it returns None
        """
        segment_index = self._check_segment_index(segment_index)
        if self.has_recording():
            return self._recording.get_times(segment_index=segment_index)
        else:
            return None

    def _save(self, format="numpy_folder", **save_kwargs):
        """
        This function replaces the old CachesortingExtractor, but enables more engines
        for caching a results.

        Since v0.98.0 "numpy_folder" is used by defult.
        From v0.96.0 to 0.97.0 "npz_folder" was the default.

        """
        if format == "numpy_folder":
            from .sortingfolder import NumpyFolderSorting

            folder = save_kwargs.pop("folder")
            NumpyFolderSorting.write_sorting(self, folder)
            cached = NumpyFolderSorting(folder)

            if self.has_recording():
                warnings.warn("The registered recording will not be persistent on disk, but only available in memory")
                cached.register_recording(self._recording)

        elif format == "npz_folder":
            from .sortingfolder import NpzFolderSorting

            folder = save_kwargs.pop("folder")
            NpzFolderSorting.write_sorting(self, folder)
            cached = NpzFolderSorting(folder_path=folder)

            if self.has_recording():
                warnings.warn("The registered recording will not be persistent on disk, but only available in memory")
                cached.register_recording(self._recording)

        elif format == "memory":
            from .numpyextractors import NumpySorting

            cached = NumpySorting.from_sorting(self)
        else:
            raise ValueError(f"format {format} not supported")
        return cached

    def get_unit_property(self, unit_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(unit_id)]
        return v

    def get_total_num_spikes(self):
        warnings.warn(
            "Sorting.get_total_num_spikes() is deprecated, se sorting.count_num_spikes_per_unit()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.count_num_spikes_per_unit()

    def count_num_spikes_per_unit(self) -> dict:
        """
        For each unit : get number of spikes  across segments.

        Returns
        -------
        dict
            Dictionary with unit_ids as key and number of spikes as values
        """
        num_spikes = {}

        if self._cached_spike_trains is not None:
            for unit_id in self.unit_ids:
                n = 0
                for segment_index in range(self.get_num_segments()):
                    st = self.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                    n += st.size
                num_spikes[unit_id] = n
        else:
            spike_vector = self.to_spike_vector()
            unit_indices, counts = np.unique(spike_vector["unit_index"], return_counts=True)
            for unit_index, unit_id in enumerate(self.unit_ids):
                if unit_index in unit_indices:
                    idx = np.argmax(unit_indices == unit_index)
                    num_spikes[unit_id] = counts[idx]
                else:  # This unit has no spikes, hence it's not in the counts array.
                    num_spikes[unit_id] = 0

        return num_spikes

    def count_total_num_spikes(self):
        """
        Get total number of spikes summed across segment and units.

        Returns
        -------
        total_num_spikes: int
            The total number of spike
        """
        return self.to_spike_vector().size

    def select_units(self, unit_ids, renamed_unit_ids=None):
        """
        Selects a subset of units

        Parameters
        ----------
        unit_ids : numpy.array or list
            List of unit ids to keep
        renamed_unit_ids : numpy.array or list, default: None
            If given, the kept unit ids are renamed

        Returns
        -------
        BaseSorting
            Sorting object with selected units
        """
        from spikeinterface import UnitsSelectionSorting

        sub_sorting = UnitsSelectionSorting(self, unit_ids, renamed_unit_ids=renamed_unit_ids)
        return sub_sorting

    def remove_units(self, remove_unit_ids):
        """
        Removes a subset of units

        Parameters
        ----------
        remove_unit_ids :  numpy.array or list
            List of unit ids to remove

        Returns
        -------
        BaseSorting
            Sorting object without removed units
        """
        from spikeinterface import UnitsSelectionSorting

        new_unit_ids = self.unit_ids[~np.isin(self.unit_ids, remove_unit_ids)]
        new_sorting = UnitsSelectionSorting(self, new_unit_ids)
        return new_sorting

    def remove_empty_units(self):
        """
        Removes units with empty spike trains

        Returns
        -------
        BaseSorting
            Sorting object with non-empty units
        """
        non_empty_units = self.get_non_empty_unit_ids()
        return self.select_units(non_empty_units)

    def get_non_empty_unit_ids(self):
        non_empty_units = []
        for segment_index in range(self.get_num_segments()):
            for unit in self.get_unit_ids():
                if len(self.get_unit_spike_train(unit, segment_index=segment_index)) > 0:
                    non_empty_units.append(unit)
        non_empty_units = np.unique(non_empty_units)
        return non_empty_units

    def get_empty_unit_ids(self):
        unit_ids = self.get_unit_ids()
        empty_units = unit_ids[~np.isin(unit_ids, self.get_non_empty_unit_ids())]
        return empty_units

    def frame_slice(self, start_frame, end_frame, check_spike_frames=True):
        from spikeinterface import FrameSliceSorting

        sub_sorting = FrameSliceSorting(
            self, start_frame=start_frame, end_frame=end_frame, check_spike_frames=check_spike_frames
        )
        return sub_sorting

    def get_all_spike_trains(self, outputs="unit_id"):
        """
        Return all spike trains concatenated.

        This is deprecated use  sorting.to_spike_vector() instead
        """

        warnings.warn(
            "Sorting.get_all_spike_trains() will be deprecated. Sorting.to_spike_vector() instead",
            DeprecationWarning,
            stacklevel=2,
        )

        assert outputs in ("unit_id", "unit_index")
        spikes = []
        for segment_index in range(self.get_num_segments()):
            spike_times = []
            spike_labels = []
            for i, unit_id in enumerate(self.unit_ids):
                st = self.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                spike_times.append(st)
                if outputs == "unit_id":
                    spike_labels.append(np.array([unit_id] * st.size))
                elif outputs == "unit_index":
                    spike_labels.append(np.zeros(st.size, dtype="int64") + i)

            if len(spike_times) > 0:
                spike_times = np.concatenate(spike_times)
                spike_labels = np.concatenate(spike_labels)
                order = np.argsort(spike_times)
                spike_times = spike_times[order]
                spike_labels = spike_labels[order]
            else:
                spike_times = np.array([], dtype=np.int64)
                spike_labels = np.array([], dtype=np.int64)

            spikes.append((spike_times, spike_labels))
        return spikes

    def to_spike_vector(self, concatenated=True, extremum_channel_inds=None, use_cache=True):
        """
        Construct a unique structured numpy vector concatenating all spikes
        with several fields: sample_index, unit_index, segment_index.

        See also `get_all_spike_trains()`

        Parameters
        ----------
        concatenated: bool, default: True
            With concatenated=True the output is one numpy "spike vector" with spikes from all segments.
            With concatenated=False the output is a list "spike vector" by segment.
        extremum_channel_inds: None or dict, default: None
            If a dictionnary of unit_id to channel_ind is given then an extra field "channel_index".
            This can be convinient for computing spikes postion after sorter.
            This dict can be computed with `get_template_extremum_channel(we, outputs="index")`
        use_cache: bool, default: True
            When True the spikes vector is cached as an attribute of the object (`_cached_spike_vector`).
            This caching only occurs when extremum_channel_inds=None.

        Returns
        -------
        spikes: np.array
            Structured numpy array ("sample_index", "unit_index", "segment_index") with all spikes
            Or ("sample_index", "unit_index", "segment_index", "channel_index") if extremum_channel_inds
            is given

        """

        spike_dtype = minimum_spike_dtype
        if extremum_channel_inds is not None:
            spike_dtype = spike_dtype + [("channel_index", "int64")]
            ext_channel_inds = np.array([extremum_channel_inds[unit_id] for unit_id in self.unit_ids])

        if use_cache and self._cached_spike_vector is not None:
            # the cache already exists
            if extremum_channel_inds is None:
                spikes = self._cached_spike_vector
            else:
                spikes = np.zeros(self._cached_spike_vector.size, dtype=spike_dtype)
                spikes["sample_index"] = self._cached_spike_vector["sample_index"]
                spikes["unit_index"] = self._cached_spike_vector["unit_index"]
                spikes["segment_index"] = self._cached_spike_vector["segment_index"]
                if extremum_channel_inds is not None:
                    spikes["channel_index"] = ext_channel_inds[spikes["unit_index"]]

            if not concatenated:
                spikes_ = []
                for segment_index in range(self.get_num_segments()):
                    s0, s1 = np.searchsorted(spikes["segment_index"], [segment_index, segment_index + 1], side="left")
                    spikes_.append(spikes[s0:s1])
                spikes = spikes_

        else:
            # the cache not needed or do not exists yet
            spikes = []
            for segment_index in range(self.get_num_segments()):
                sample_indices = []
                unit_indices = []
                for u, unit_id in enumerate(self.unit_ids):
                    spike_times = self.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                    sample_indices.append(spike_times)
                    unit_indices.append(np.full(spike_times.size, u, dtype="int64"))

                if len(sample_indices) > 0:
                    sample_indices = np.concatenate(sample_indices, dtype="int64")
                    unit_indices = np.concatenate(unit_indices, dtype="int64")
                    order = np.argsort(sample_indices)
                    sample_indices = sample_indices[order]
                    unit_indices = unit_indices[order]

                spikes_in_seg = np.zeros(len(sample_indices), dtype=spike_dtype)
                spikes_in_seg["sample_index"] = sample_indices
                spikes_in_seg["unit_index"] = unit_indices
                spikes_in_seg["segment_index"] = segment_index
                if extremum_channel_inds is not None:
                    # vector way
                    spikes_in_seg["channel_index"] = ext_channel_inds[spikes_in_seg["unit_index"]]
                spikes.append(spikes_in_seg)

            if concatenated:
                spikes = np.concatenate(spikes)

            if use_cache and self._cached_spike_vector is None and extremum_channel_inds is None:
                # cache it if necessary but only without "channel_index"
                if concatenated:
                    self._cached_spike_vector = spikes
                else:
                    self._cached_spike_vector = np.concatenate(spikes)

        return spikes

    def to_numpy_sorting(self, propagate_cache=True):
        """
        Turn any sorting in a NumpySorting.
        useful to have it in memory with a unique vector representation.

        Parameters
        ----------
        propagate_cache : bool
            Propagate the cache of indivudual spike trains.

        """
        from .numpyextractors import NumpySorting

        sorting = NumpySorting.from_sorting(self)
        if propagate_cache and self._cached_spike_trains is not None:
            sorting._cached_spike_trains = self._cached_spike_trains
        return sorting

    def to_shared_memory_sorting(self):
        """
        Turn any sorting in a SharedMemorySorting.
        Usefull to have it in memory with a unique vector representation and sharable across processes.
        """
        from .numpyextractors import SharedMemorySorting

        sorting = SharedMemorySorting.from_sorting(self)
        return sorting

    def to_multiprocessing(self, n_jobs):
        """
        When necessary turn sorting object into:
        * NumpySorting when n_jobs=1
        * SharedMemorySorting when n_jobs>1

        If the sorting is already NumpySorting, SharedMemorySorting or NumpyFolderSorting
        then this return the sortign itself, no transformation so.

        Parameters
        ----------
        n_jobs: int
            The number of jobs.
        Returns
        -------
        sharable_sorting:
            A sorting that can be used for multiprocessing.
        """
        from .numpyextractors import NumpySorting, SharedMemorySorting
        from .sortingfolder import NumpyFolderSorting

        if n_jobs == 1:
            if isinstance(self, (NumpySorting, SharedMemorySorting, NumpyFolderSorting)):
                return self
            else:
                return NumpySorting.from_sorting(self)
        else:
            if isinstance(self, (SharedMemorySorting, NumpyFolderSorting)):
                return self
            else:
                return SharedMemorySorting.from_sorting(self)


class BaseSortingSegment(BaseSegment):
    """
    Abstract class representing several units and relative spiketrain inside a segment.
    """

    def __init__(self, t_start=None):
        self._t_start = t_start
        BaseSegment.__init__(self)

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        """Get the spike train for a unit.

        Parameters
        ----------
        unit_id
        start_frame: int, default: None
        end_frame: int, default: None

        Returns
        -------
        np.ndarray

        """
        # must be implemented in subclass
        raise NotImplementedError
