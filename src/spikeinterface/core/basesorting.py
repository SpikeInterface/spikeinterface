from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

from .base import BaseExtractor, BaseSegment
from .waveform_tools import has_exceeding_spikes


minimum_spike_dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]


class BaseSorting(BaseExtractor):
    """
    Abstract class representing several segment several units and relative spiketrains.
    """

    def __init__(self, sampling_frequency: float, unit_ids: list):
        BaseExtractor.__init__(self, unit_ids)
        self._sampling_frequency = float(sampling_frequency)
        self._sorting_segments: list[BaseSortingSegment] = []
        # this weak link is to handle times from a recording object
        self._recording = None
        self._sorting_info = None

        # caching
        self._cached_spike_vector = None
        self._cached_spike_trains = {}

    def __repr__(self):
        return self._repr_header()

    def _repr_header(self, display_name=True):
        nseg = self.get_num_segments()
        nunits = self.get_num_units()
        sf_khz = self.get_sampling_frequency() / 1000.0
        if display_name and self.name != self.__class__.__name__:
            name = f"{self.name} ({self.__class__.__name__})"
        else:
            name = self.__class__.__name__
        txt = f"{name}: {nunits} units - {nseg} segments - {sf_khz:0.1f}kHz"
        if "file_path" in self._kwargs:
            txt += "\n  file_path: {}".format(self._kwargs["file_path"])
        return txt

    def _repr_html_(self, display_name=True):
        common_style = "margin-left: 10px;"
        border_style = "border:1px solid #ddd; padding:10px;"

        html_header = f"<div style='{border_style}'><strong>{self._repr_header(display_name)}</strong></div>"

        html_unit_ids = f"<details style='{common_style}'>  <summary><strong>Unit IDs</strong></summary><ul>"
        html_unit_ids += f"{self.unit_ids} </details>"

        html_extra = self._get_common_repr_html(common_style)

        html_repr = html_header + html_unit_ids + html_extra
        return html_repr

    @property
    def unit_ids(self):
        return self._main_ids

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def get_unit_ids(self) -> list:
        return self._main_ids

    def get_num_units(self) -> int:
        return len(self.get_unit_ids())

    def add_sorting_segment(self, sorting_segment):
        self._sorting_segments.append(sorting_segment)
        sorting_segment.set_parent_extractor(self)

    def get_sampling_frequency(self) -> float:
        return self._sampling_frequency

    def get_num_segments(self) -> int:
        return len(self._sorting_segments)

    def get_num_samples(self, segment_index=None) -> int:
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

    def get_total_samples(self) -> int:
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

    def get_total_duration(self) -> float:
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
        unit_id: str | int,
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
                    "int64", copy=False
                )
                self._cached_spike_trains[segment_index][unit_id] = spike_frames
            else:
                spike_frames = self._cached_spike_trains[segment_index][unit_id]
            if start_frame is not None:
                start = np.searchsorted(spike_frames, start_frame)
                spike_frames = spike_frames[start:]
            if end_frame is not None:
                end = np.searchsorted(spike_frames, end_frame)
                spike_frames = spike_frames[:end]
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
        """
        Register a recording to the sorting. If the sorting and recording both contain
        time information, the recording’s time information will be used.

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
            if has_exceeding_spikes(self, recording):
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

    def has_recording(self) -> bool:
        return self._recording is not None

    def has_time_vector(self, segment_index=None) -> bool:
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

        elif format == "zarr":
            from .zarrextractors import ZarrSortingExtractor

            zarr_path = save_kwargs.pop("zarr_path")
            storage_options = save_kwargs.pop("storage_options")
            ZarrSortingExtractor.write_sorting(self, zarr_path, storage_options, **save_kwargs)
            cached = ZarrSortingExtractor(zarr_path, storage_options)

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
            if save_kwargs.get("sharedmem", True):
                from .numpyextractors import SharedMemorySorting

                cached = SharedMemorySorting.from_sorting(self)
            else:
                from .numpyextractors import NumpySorting

                cached = NumpySorting.from_sorting(self)
        else:
            raise ValueError(f"format {format} not supported")
        return cached

    def get_unit_property(self, unit_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(unit_id)]
        return v

    def count_num_spikes_per_unit(self, outputs="dict"):
        """
        For each unit : get number of spikes  across segments.

        Parameters
        ----------
        outputs : "dict" | "array", default: "dict"
            Control the type of the returned object : a dict (keys are unit_ids) or an numpy array.

        Returns
        -------
        dict or numpy.array
            Dict : Dictionary with unit_ids as key and number of spikes as values
            Numpy array : array of size len(unit_ids) in the same order as unit_ids.
        """
        num_spikes = np.zeros(self.unit_ids.size, dtype="int64")

        # speed strategy by order
        # 1. if _cached_spike_trains have all units then use it
        # 2. if _cached_spike_vector is not non use it
        # 3. loop with get_unit_spike_train

        # check if all spiketrains are cached
        if len(self._cached_spike_trains) == self.get_num_segments():
            all_spiketrain_are_cached = True
            for segment_index in range(self.get_num_segments()):
                if len(self._cached_spike_trains[segment_index]) != self.unit_ids.size:
                    all_spiketrain_are_cached = False
                    break
        else:
            all_spiketrain_are_cached = False

        if all_spiketrain_are_cached or self._cached_spike_vector is None:
            # case 1 or 3
            for unit_index, unit_id in enumerate(self.unit_ids):
                for segment_index in range(self.get_num_segments()):
                    st = self.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                    num_spikes[unit_index] += st.size
        elif self._cached_spike_vector is not None:
            # case 2
            spike_vector = self.to_spike_vector()
            unit_indices, counts = np.unique(spike_vector["unit_index"], return_counts=True)
            num_spikes[unit_indices] = counts

        if outputs == "array":
            return num_spikes
        elif outputs == "dict":
            num_spikes = dict(zip(self.unit_ids, num_spikes))
            return num_spikes
        else:
            raise ValueError("count_num_spikes_per_unit() output must be 'dict' or 'array'")

    def count_total_num_spikes(self) -> int:
        """
        Get total number of spikes in the sorting.

        This is the sum of all spikes in all segments across all units.

        Returns
        -------
        total_num_spikes : int
            The total number of spike
        """
        return self.to_spike_vector().size

    def select_units(self, unit_ids, renamed_unit_ids=None) -> BaseSorting:
        """
        Returns a new sorting object which contains only a selected subset of units.


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

    def rename_units(self, new_unit_ids: np.ndarray | list) -> BaseSorting:
        """
        Returns a new sorting object with renamed units.


        Parameters
        ----------
        new_unit_ids : numpy.array or list
            List of new names for unit ids.
            They should map positionally to the existing unit ids.

        Returns
        -------
        BaseSorting
            Sorting object with renamed units
        """
        from spikeinterface import UnitsSelectionSorting

        sub_sorting = UnitsSelectionSorting(self, renamed_unit_ids=new_unit_ids)
        return sub_sorting

    def remove_units(self, remove_unit_ids) -> BaseSorting:
        """
        Returns a new sorting object with contains only a selected subset of units.

        Parameters
        ----------
        remove_unit_ids :  numpy.array or list
            List of unit ids to remove

        Returns
        -------
        BaseSorting
            Sorting without the removed units
        """
        from spikeinterface import UnitsSelectionSorting

        new_unit_ids = self.unit_ids[~np.isin(self.unit_ids, remove_unit_ids)]
        new_sorting = UnitsSelectionSorting(self, new_unit_ids)
        return new_sorting

    def remove_empty_units(self):
        """
        Returns a new sorting object which contains only units with at least one spike.
        For multi-segments, a unit is considered empty if it contains no spikes in all segments.

        Returns
        -------
        BaseSorting
            Sorting object with non-empty units
        """
        non_empty_units = self.get_non_empty_unit_ids()
        return self.select_units(non_empty_units)

    def get_non_empty_unit_ids(self) -> np.ndarray:
        """
        Return the unit IDs that have at least one spike across all segments.

        This method computes the number of spikes for each unit using
        `count_num_spikes_per_unit` and filters out units with zero spikes.

        Returns
        -------
        np.ndarray
            Array of unit IDs (same dtype as self.unit_ids) for which at least one spike exists.
        """
        num_spikes_per_unit = self.count_num_spikes_per_unit()

        return np.array([unit_id for unit_id in self.unit_ids if num_spikes_per_unit[unit_id] != 0])

    def get_empty_unit_ids(self) -> np.ndarray:
        """
        Return the unit IDs that have zero spikes across all segments.

        This method returns the complement of `get_non_empty_unit_ids` with respect
        to all unit IDs in the sorting.

        Returns
        -------
        np.ndarray
            Array of unit IDs (same dtype as self.unit_ids) for which no spikes exist.
        """
        unit_ids = self.unit_ids
        empty_units = unit_ids[~np.isin(unit_ids, self.get_non_empty_unit_ids())]
        return empty_units

    def frame_slice(self, start_frame, end_frame, check_spike_frames=True):
        from spikeinterface import FrameSliceSorting

        sub_sorting = FrameSliceSorting(
            self, start_frame=start_frame, end_frame=end_frame, check_spike_frames=check_spike_frames
        )
        return sub_sorting

    def time_slice(self, start_time: float | None, end_time: float | None) -> BaseSorting:
        """
        Returns a new sorting object, restricted to the time interval [start_time, end_time].

        Parameters
        ----------
        start_time : float | None, default: None
            The start time in seconds. If not provided it is set to 0.
        end_time : float | None, default: None
            The end time in seconds. If not provided it is set to the total duration.

        Returns
        -------
        BaseSorting
            A new sorting object with only samples between start_time and end_time
        """

        assert self.get_num_segments() == 1, "Time slicing is only supported for single segment sortings."

        start_frame = self.time_to_sample_index(start_time, segment_index=0) if start_time else None
        end_frame = self.time_to_sample_index(end_time, segment_index=0) if end_time else None

        return self.frame_slice(start_frame=start_frame, end_frame=end_frame)

    def time_to_sample_index(self, time, segment_index=0):
        """
        Transform time in seconds into sample index
        """
        if self.has_recording():
            sample_index = self._recording.time_to_sample_index(time, segment_index=segment_index)
        else:
            segment = self._sorting_segments[segment_index]
            t_start = segment._t_start if segment._t_start is not None else 0
            sample_index = round((time - t_start) * self.get_sampling_frequency())

        return sample_index

    def precompute_spike_trains(self, from_spike_vector=None):
        """
        Pre-computes and caches all spike trains for this sorting

        Parameters
        ----------
        from_spike_vector : None | bool, default: None
            If None, then it is automatic depending on whether the spike vector is cached.
            If True, will compute it from the spike vector.
            If False, will call `get_unit_spike_train` for each segment for each unit.
        """
        from .sorting_tools import spike_vector_to_spike_trains

        unit_ids = self.unit_ids

        if from_spike_vector is None:
            # if spike vector is cached then use it
            from_spike_vector = self._cached_spike_vector is not None

        if from_spike_vector:
            self._cached_spike_trains = spike_vector_to_spike_trains(self.to_spike_vector(concatenated=False), unit_ids)

        else:
            for segment_index in range(self.get_num_segments()):
                for unit_id in unit_ids:
                    self.get_unit_spike_train(unit_id, segment_index=segment_index, use_cache=True)

    def _custom_cache_spike_vector(self) -> None:
        """
        Function that can be implemented by some children sorting to quickly
        cache the spike vector without computing it from spike trains
        (e.g. computing it from a sorting parent).
        This function should set the `self._cached_spike_vector`, see for
        instance the `UnitsSelectionSorting` implementation.
        """
        pass

    def to_spike_vector(
        self, concatenated=True, extremum_channel_inds=None, use_cache=True
    ) -> np.ndarray | list[np.ndarray]:
        """
        Construct a unique structured numpy vector concatenating all spikes
        with several fields: sample_index, unit_index, segment_index.


        Parameters
        ----------
        concatenated : bool, default: True
            With concatenated=True the output is one numpy "spike vector" with spikes from all segments.
            With concatenated=False the output is a list "spike vector" by segment.
        extremum_channel_inds : None or dict, default: None
            If a dictionnary of unit_id to channel_ind is given then an extra field "channel_index".
            This can be convinient for computing spikes postion after sorter.
            This dict can be computed with `get_template_extremum_channel(we, outputs="index")`
        use_cache : bool, default: True
            When True the spikes vector is cached as an attribute of the object (`_cached_spike_vector`).
            This caching only occurs when extremum_channel_inds=None.

        Returns
        -------
        spikes : np.array
            Structured numpy array ("sample_index", "unit_index", "segment_index") with all spikes
            Or ("sample_index", "unit_index", "segment_index", "channel_index") if extremum_channel_inds
            is given

        """

        spike_dtype = minimum_spike_dtype
        if extremum_channel_inds is not None:
            spike_dtype = spike_dtype + [("channel_index", "int64")]
            ext_channel_inds = np.array([extremum_channel_inds[unit_id] for unit_id in self.unit_ids])

        if use_cache and self._cached_spike_vector is None:
            self._custom_cache_spike_vector()

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
        n_jobs : int
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
        start_frame : int, default: None
        end_frame : int, default: None

        Returns
        -------
        np.ndarray

        """
        # must be implemented in subclass
        raise NotImplementedError


class SpikeVectorSortingSegment(BaseSortingSegment):
    """
    A sorting segment that stores spike times as a spike vector.
    """

    def __init__(self, spikes, segment_index, unit_ids):
        BaseSortingSegment.__init__(self)
        self.spikes = spikes
        self.segment_index = segment_index
        self.unit_ids = list(unit_ids)
        self.spikes_in_seg = None

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        if self.spikes_in_seg is None:
            # the slicing of segment is done only once the first time
            # this fasten the constructor a lot
            s0, s1 = np.searchsorted(self.spikes["segment_index"], [self.segment_index, self.segment_index + 1])
            self.spikes_in_seg = self.spikes[s0:s1]

        start = 0 if start_frame is None else np.searchsorted(self.spikes_in_seg["sample_index"], start_frame)
        end = (
            len(self.spikes_in_seg)
            if end_frame is None
            else np.searchsorted(self.spikes_in_seg["sample_index"], end_frame)
        )

        unit_index = self.unit_ids.index(unit_id)
        times = self.spikes_in_seg[start:end][self.spikes_in_seg[start:end]["unit_index"] == unit_index]["sample_index"]

        return times
