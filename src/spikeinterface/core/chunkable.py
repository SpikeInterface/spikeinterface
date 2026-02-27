from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import warnings

import numpy as np

from spikeinterface.core.base import BaseExtractor, BaseSegment


class ChunkableMixin(ABC):
    """
    Abstract mixin class for chunkable objects. Note that the mixin can only be used
    for classes that inherit from BaseExtractor.
    Provides methods to handle chunked data access, that can be used for parallelization.
    In addition, since chunkable objects are continuous data, time handling methods are provided.

    The Mixin is abstract since all methods need to be implemented in the child class in order
    for it to function properly.
    """

    _preferred_mp_context = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, BaseExtractor):
            raise TypeError(f"{cls.__name__} must inherit from BaseExtractor to use Chunkable mixin.")

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_num_samples(self, segment_index: int | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_sample_size_in_bytes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_shape(self, segment_index: int | None = None) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def get_data(self, start_frame: int, end_frame: int, segment_index: int | None = None, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _extra_copy_metadata(self, other: "ChunkableMixin", **kwargs) -> None:
        """
        Copy metadata from another Chunkable object.

        Parameters
        ----------
        other : ChunkableMixin
            The object from which to copy metadata.
        """
        # inherit preferred mp context if any
        if self.__class__._preferred_mp_context is not None:
            other.__class__._preferred_mp_context = self.__class__._preferred_mp_context

    def get_preferred_mp_context(self):
        """
        Get the preferred context for multiprocessing.
        If None, the context is set by the multiprocessing package.
        """
        return self.__class__._preferred_mp_context

    def get_memory_size(self, segment_index=None) -> int:
        """
        Returns the memory size of segment_index in bytes.

        Parameters
        ----------
        segment_index : int or None, default: None
            The index of the segment for which the memory size should be calculated.
            For multi-segment objects, it is required, default: None
            With single segment recording returns the memory size of the single segment

        Returns
        -------
        int
            The memory size of the specified segment in bytes.
        """
        segment_index = self._check_segment_index(segment_index)
        num_samples = self.get_num_samples(segment_index=segment_index)
        sample_size_in_bytes = self.get_sample_size_in_bytes()

        memory_bytes = num_samples * sample_size_in_bytes

        return memory_bytes

    def get_total_memory_size(self) -> int:
        """
        Returns the sum in bytes of all the memory sizes of the segments.

        Returns
        -------
        int
            The total memory size in bytes for all segments.
        """
        memory_per_segment = (self.get_memory_size(segment_index) for segment_index in range(self.get_num_segments()))
        return sum(memory_per_segment)

    # Add time handling
    def get_time_info(self, segment_index=None) -> dict:
        """
        Retrieves the timing attributes for a given segment index. As with
        other recorders this method only needs a segment index in the case
        of multi-segment recordings.

        Returns
        -------
        dict
            A dictionary containing the following key-value pairs:

            - "sampling_frequency" : The sampling frequency of the RecordingSegment.
            - "t_start" : The start time of the RecordingSegment.
            - "time_vector" : The time vector of the RecordingSegment.

        Notes
        -----
        The keys are always present, but the values may be None.
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        time_kwargs = rs.get_times_kwargs()

        return time_kwargs

    def get_times(self, segment_index=None, start_frame=None, end_frame=None) -> np.ndarray:
        """Get time vector for a recording segment.

        If the segment has a time_vector, then it is returned. Otherwise
        a time_vector is constructed on the fly with sampling frequency.
        If t_start is defined and the time vector is constructed on the fly,
        the first time will be t_start. Otherwise it will start from 0.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)
        start_frame : int or None, default: None
            The start frame for the time vector
        end_frame : int or None, default: None
            The end frame for the time vector

        Returns
        -------
        np.array
            The 1d times array
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        times = rs.get_times(start_frame=start_frame, end_frame=end_frame)
        return times

    def get_start_time(self, segment_index=None) -> float:
        """Get the start time of the recording segment.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        float
            The start time in seconds
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        return rs.get_start_time()

    def get_end_time(self, segment_index=None) -> float:
        """Get the stop time of the recording segment.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        float
            The stop time in seconds
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        return rs.get_end_time()

    def has_time_vector(self, segment_index: Optional[int] = None):
        """Check if the segment of the recording has a time vector.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        bool
            True if the recording has time vectors, False otherwise
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        d = rs.get_times_kwargs()
        return d["time_vector"] is not None

    def set_times(self, times, segment_index=None, with_warning=True):
        """Set times for a recording segment.

        Parameters
        ----------
        times : 1d np.array
            The time vector
        segment_index : int or None, default: None
            The segment index (required for multi-segment)
        with_warning : bool, default: True
            If True, a warning is printed
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]

        assert times.ndim == 1, "Time must have ndim=1"
        assert rs.get_num_samples() == times.shape[0], "times have wrong shape"

        rs.t_start = None
        rs.time_vector = times.astype("float64", copy=False)

        if with_warning:
            warnings.warn(
                "Setting times with Recording.set_times() is not recommended because "
                "times are not always propagated across preprocessing"
                "Use this carefully!"
            )

    def reset_times(self):
        """
        Reset time information in-memory for all segments that have a time vector.
        If the timestamps come from a file, the files won't be modified. but only the in-memory
        attributes of the recording objects are deleted. Also `t_start` is set to None and the
        segment's sampling frequency is set to the recording's sampling frequency.
        """
        for segment_index in range(self.get_num_segments()):
            rs = self.segments[segment_index]
            if self.has_time_vector(segment_index):
                rs.time_vector = None
            rs.t_start = None
            rs.sampling_frequency = self.sampling_frequency

    def shift_times(self, shift: int | float, segment_index: int | None = None) -> None:
        """
        Shift all times by a scalar value.

        Parameters
        ----------
        shift : int | float
            The shift to apply. If positive, times will be increased by `shift`.
            e.g. shifting by 1 will be like the recording started 1 second later.
            If negative, the start time will be decreased i.e. as if the recording
            started earlier.

        segment_index : int | None
            The segment on which to shift the times.
            If `None`, all segments will be shifted.
        """
        if segment_index is None:
            segments_to_shift = range(self.get_num_segments())
        else:
            segments_to_shift = (segment_index,)

        for segment_index in segments_to_shift:
            rs = self.segments[segment_index]

            if self.has_time_vector(segment_index=segment_index):
                rs.time_vector += shift
            else:
                new_start_time = 0 + shift if rs.t_start is None else rs.t_start + shift
                rs.t_start = new_start_time

    def sample_index_to_time(self, sample_ind, segment_index=None):
        """
        Transform sample index into time in seconds
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        return rs.sample_index_to_time(sample_ind)

    def time_to_sample_index(self, time_s, segment_index=None):
        """
        Transform time in seconds into sample index
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self.segments[segment_index]
        return rs.time_to_sample_index(time_s)

    def get_total_samples(self) -> int:
        """
        Returns the sum of the number of samples in each segment.

        Returns
        -------
        int
            The total number of samples
        """
        num_segments = self.get_num_segments()
        samples_per_segment = (self.get_num_samples(segment_index) for segment_index in range(num_segments))

        return sum(samples_per_segment)

    def get_duration(self, segment_index=None) -> float:
        """
        Returns the duration in seconds.

        Parameters
        ----------
        segment_index : int or None, default: None
            The sample index to retrieve the duration for.
            For multi-segment objects, it is required, default: None
            With single segment recording returns the duration of the single segment

        Returns
        -------
        float
            The duration in seconds
        """
        segment_duration = (
            self.get_end_time(segment_index) - self.get_start_time(segment_index) + (1 / self.get_sampling_frequency())
        )
        return segment_duration

    def get_total_duration(self) -> float:
        """
        Returns the total duration in seconds

        Returns
        -------
        float
            The duration in seconds
        """
        duration = sum([self.get_duration(segment_index) for segment_index in range(self.get_num_segments())])
        return duration

    def _get_t_starts(self):
        # handle t_starts
        t_starts = []
        for rs in self.segments:
            d = rs.get_times_kwargs()
            t_starts.append(d["t_start"])

        if all(t_start is None for t_start in t_starts):
            t_starts = None
        return t_starts

    def _get_time_vectors(self):
        time_vectors = []
        for rs in self.segments:
            d = rs.get_times_kwargs()
            time_vectors.append(d["time_vector"])
        if all(time_vector is None for time_vector in time_vectors):
            time_vectors = None
        return time_vectors


class ChunkableSegment(BaseSegment):
    """Class for chunkable segments, which provide methods to handle time kwargs."""

    def __init__(self, sampling_frequency=None, t_start=None, time_vector=None):
        # sampling_frequency and time_vector are exclusive
        if sampling_frequency is None:
            assert time_vector is not None, "Pass either 'sampling_frequency' or 'time_vector'"
            assert time_vector.ndim == 1, "time_vector should be a 1D array"

        if time_vector is None:
            assert sampling_frequency is not None, "Pass either 'sampling_frequency' or 'time_vector'"

        self.sampling_frequency = sampling_frequency
        self.t_start = t_start
        self.time_vector = time_vector

        BaseSegment.__init__(self)

    def get_times(self, start_frame: int | None = None, end_frame: int | None = None) -> np.ndarray:
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        if self.time_vector is not None:
            self.time_vector = np.asarray(self.time_vector)
            return self.time_vector[start_frame:end_frame]
        else:
            time_vector = np.arange(start_frame, end_frame, dtype="float64")
            time_vector /= self.sampling_frequency
            if self.t_start is not None:
                time_vector += self.t_start
            return time_vector

    def get_start_time(self) -> float:
        if self.time_vector is not None:
            return self.time_vector[0]
        else:
            return self.t_start if self.t_start is not None else 0.0

    def get_end_time(self) -> float:
        if self.time_vector is not None:
            return self.time_vector[-1]
        else:
            t_stop = (self.get_num_samples() - 1) / self.sampling_frequency
            if self.t_start is not None:
                t_stop += self.t_start
            return t_stop

    def get_times_kwargs(self) -> dict:
        """
        Retrieves the timing attributes characterizing a RecordingSegment

        Returns
        -------
        dict
            A dictionary containing the following key-value pairs:

            - "sampling_frequency" : The sampling frequency of the RecordingSegment.
            - "t_start" : The start time of the RecordingSegment.
            - "time_vector" : The time vector of the RecordingSegment.

        Notes
        -----
        The keys are always present, but the values may be None.
        """
        time_kwargs = dict(
            sampling_frequency=self.sampling_frequency, t_start=self.t_start, time_vector=self.time_vector
        )
        return time_kwargs

    def sample_index_to_time(self, sample_ind):
        """
        Transform sample index into time in seconds
        """
        if self.time_vector is None:
            time_s = sample_ind / self.sampling_frequency
            if self.t_start is not None:
                time_s += self.t_start
        else:
            time_s = self.time_vector[sample_ind]
        return time_s

    def time_to_sample_index(self, time_s):
        """
        Transform time in seconds into sample index
        """
        if self.time_vector is None:
            if self.t_start is None:
                sample_index = time_s * self.sampling_frequency
            else:
                sample_index = (time_s - self.t_start) * self.sampling_frequency
            sample_index = np.round(sample_index).astype(np.int64)
        else:
            sample_index = np.searchsorted(self.time_vector, time_s, side="right") - 1

        return sample_index

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal segment

        Returns:
            SampleIndex : Number of samples in the signal segment
        """
        # must be implemented in subclass
        raise NotImplementedError
