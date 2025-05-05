from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from probeinterface import Probe, ProbeGroup, read_probeinterface, select_axes, write_probeinterface

from .base import BaseSegment
from .baserecordingsnippets import BaseRecordingSnippets
from .core_tools import convert_bytes_to_str, convert_seconds_to_str
from .job_tools import split_job_kwargs
from .recording_tools import write_binary_recording


class BaseRecording(BaseRecordingSnippets):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """

    _main_annotations = BaseRecordingSnippets._main_annotations + ["is_filtered"]
    _main_properties = ["group", "location", "gain_to_uV", "offset_to_uV"]
    _main_features = []  # recording do not handle features

    _skip_properties = [
        "noise_level_std_raw",
        "noise_level_std_scaled",
        "noise_level_mad_raw",
        "noise_level_mad_scaled",
    ]

    def __init__(self, sampling_frequency: float, channel_ids: list, dtype):
        BaseRecordingSnippets.__init__(
            self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype
        )

        self._recording_segments: list[BaseRecordingSegment] = []

        # initialize main annotation and properties
        self.annotate(is_filtered=False)

    def __repr__(self):
        num_segments = self.get_num_segments()

        txt = self._repr_header()

        # Split if too long
        if len(txt) > 100:
            split_index = txt.rfind("-", 0, 100)  # Find the last "-" before character 100
            if split_index != -1:
                first_line = txt[:split_index]
                recording_string_space = len(self.name) + 2  # Length of self.name plus ": "
                white_space_to_align_with_first_line = " " * recording_string_space
                second_line = white_space_to_align_with_first_line + txt[split_index + 1 :].lstrip()
                txt = first_line + "\n" + second_line

        # Add segments info for multisegment
        if num_segments > 1:
            samples_per_segment = [self.get_num_samples(segment_index) for segment_index in range(num_segments)]
            memory_per_segment_bytes = (self.get_memory_size(segment_index) for segment_index in range(num_segments))
            durations = [self.get_duration(segment_index) for segment_index in range(num_segments)]

            samples_per_segment_formated = [f"{samples:,}" for samples in samples_per_segment]
            durations_per_segment_formated = [convert_seconds_to_str(d) for d in durations]
            memory_per_segment_formated = [convert_bytes_to_str(mem) for mem in memory_per_segment_bytes]

            def list_to_string(lst, max_size=6):
                """Add elipsis ... notation in the middle if recording has more than six segments"""
                if len(lst) <= max_size:
                    return " | ".join(x for x in lst)
                else:
                    half = max_size // 2
                    return " | ".join(x for x in lst[:half]) + " | ... | " + " | ".join(x for x in lst[-half:])

            txt += (
                f"\n"
                f"Segments:"
                f"\nSamples:   {list_to_string(samples_per_segment_formated)}"
                f"\nDurations: {list_to_string(durations_per_segment_formated)}"
                f"\nMemory:    {list_to_string(memory_per_segment_formated)}"
            )

        # Display where path from where recording was loaded
        if "file_paths" in self._kwargs:
            txt += f"\n  file_paths: {self._kwargs['file_paths']}"
        if "file_path" in self._kwargs:
            txt += f"\n  file_path: {self._kwargs['file_path']}"

        return txt

    def _repr_header(self, display_name=True):
        num_segments = self.get_num_segments()
        num_channels = self.get_num_channels()
        dtype = self.get_dtype()

        total_samples = self.get_total_samples()
        total_duration = self.get_total_duration()
        total_memory_size = self.get_total_memory_size()

        sf_hz = self.get_sampling_frequency()
        if not sf_hz.is_integer():
            sampling_frequency_repr = f"{sf_hz:f} Hz"
        else:
            # Khz for high sampling rate and Hz for LFP
            sampling_frequency_repr = f"{(sf_hz/1000.0):0.1f}kHz" if sf_hz > 10_000.0 else f"{sf_hz:0.1f}Hz"

        if display_name and self.name != self.__class__.__name__:
            name = f"{self.name} ({self.__class__.__name__})"
        else:
            name = self.__class__.__name__

        txt = (
            f"{name}: "
            f"{num_channels} channels - "
            f"{sampling_frequency_repr} - "
            f"{num_segments} segments - "
            f"{total_samples:,} samples - "
            f"{convert_seconds_to_str(total_duration)} - "
            f"{dtype} dtype - "
            f"{convert_bytes_to_str(total_memory_size)}"
        )

        return txt

    def _repr_html_(self, display_name=True):
        common_style = "margin-left: 10px;"
        border_style = "border:1px solid #ddd; padding:10px;"

        html_header = f"<div style='{border_style}'><strong>{self._repr_header(display_name)}</strong></div>"

        html_segments = ""
        if self.get_num_segments() > 1:
            html_segments += f"<details style='{common_style}'>  <summary><strong>Segments</strong></summary><ol>"
            for segment_index in range(self.get_num_segments()):
                samples = self.get_num_samples(segment_index)
                duration = self.get_duration(segment_index)
                memory_size = self.get_memory_size(segment_index)
                samples_str = f"{samples:,}"
                duration_str = convert_seconds_to_str(duration)
                memory_size_str = convert_bytes_to_str(memory_size)
                html_segments += (
                    f"<li> Samples: {samples_str}, Duration: {duration_str}, Memory: {memory_size_str}</li>"
                )

            html_segments += "</ol></details>"

        html_channel_ids = f"<details style='{common_style}'>  <summary><strong>Channel IDs</strong></summary><ul>"
        html_channel_ids += f"{self.channel_ids} </details>"

        html_extra = self._get_common_repr_html(common_style)
        html_repr = html_header + html_segments + html_channel_ids + html_extra
        return html_repr

    def get_num_segments(self) -> int:
        """
        Returns the number of segments.

        Returns
        -------
        int
            Number of segments in the recording
        """
        return len(self._recording_segments)

    def add_recording_segment(self, recording_segment):
        """Adds a recording segment.

        Parameters
        ----------
        recording_segment : BaseRecordingSegment
            The recording segment to add
        """
        # todo: check channel count and sampling frequency
        self._recording_segments.append(recording_segment)
        recording_segment.set_parent_extractor(self)

    def get_num_samples(self, segment_index: int | None = None) -> int:
        """
        Returns the number of samples for a segment.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index to retrieve the number of samples for.
            For multi-segment objects, it is required, default: None
            With single segment recording returns the number of samples in the segment

        Returns
        -------
        int
            The number of samples
        """
        segment_index = self._check_segment_index(segment_index)
        return int(self._recording_segments[segment_index].get_num_samples())

    get_num_frames = get_num_samples

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
        num_channels = self.get_num_channels()
        dtype_size_bytes = self.get_dtype().itemsize

        memory_bytes = num_samples * num_channels * dtype_size_bytes

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

    def get_traces(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_ids: list | np.array | tuple | None = None,
        order: "C" | "F" | None = None,
        return_scaled: bool = False,
        cast_unsigned: bool = False,
    ) -> np.ndarray:
        """Returns traces from recording.

        Parameters
        ----------
        segment_index : int | None, default: None
            The segment index to get traces from. If recording is multi-segment, it is required, default: None
        start_frame : int | None, default: None
            The start frame. If None, 0 is used, default: None
        end_frame : int | None, default: None
            The end frame. If None, the number of samples in the segment is used, default: None
        channel_ids : list | np.array | tuple | None, default: None
            The channel ids. If None, all channels are used, default: None
        order : "C" | "F" | None, default: None
            The order of the traces ("C" | "F"). If None, traces are returned as they are
        return_scaled : bool, default: False
            If True and the recording has scaling (gain_to_uV and offset_to_uV properties),
            traces are scaled to uV
        cast_unsigned : bool, default: False
            If True and the traces are unsigned, they are cast to integer and centered
            (an offset of (2**nbits) is subtracted)

        Returns
        -------
        np.array
            The traces (num_samples, num_channels)

        Raises
        ------
        ValueError
            If return_scaled is True, but recording does not have scaled traces
        """
        segment_index = self._check_segment_index(segment_index)
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        rs = self._recording_segments[segment_index]
        start_frame = int(start_frame) if start_frame is not None else 0
        num_samples = rs.get_num_samples()
        end_frame = int(min(end_frame, num_samples)) if end_frame is not None else num_samples
        traces = rs.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices)
        if order is not None:
            assert order in ["C", "F"]
            traces = np.asanyarray(traces, order=order)

        if cast_unsigned:
            dtype = traces.dtype
            # if dtype is unsigned, return centered signed signal
            if dtype.kind == "u":
                itemsize = dtype.itemsize
                assert itemsize < 8, "Cannot upcast uint64!"
                nbits = dtype.itemsize * 8
                # upcast to int with double itemsize
                traces = traces.astype(f"int{2 * (dtype.itemsize) * 8}") - 2 ** (nbits - 1)
                traces = traces.astype(f"int{dtype.itemsize * 8}")

        if return_scaled:
            if not self.has_scaleable_traces():
                if self._dtype.kind == "f":
                    # here we do not truely have scale but we assume this is scaled
                    # this helps a lot for simulated data
                    pass
                else:
                    raise ValueError(
                        "This recording does not support return_scaled=True (need gain_to_uV and offset_"
                        "to_uV properties)"
                    )
            else:
                gains = self.get_property("gain_to_uV")
                offsets = self.get_property("offset_to_uV")
                gains = gains[channel_indices].astype("float32", copy=False)
                offsets = offsets[channel_indices].astype("float32", copy=False)
                traces = traces.astype("float32", copy=False) * gains + offsets
        return traces

    def has_scaled_traces(self) -> bool:
        """Checks if the recording has scaled traces

        Returns
        -------
        bool
            True if the recording has scaled traces, False otherwise
        """
        warnings.warn(
            "`has_scaled_traces` is deprecated and will be removed in 0.103.0. Use has_scaleable_traces() instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.has_scaled()

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
        rs = self._recording_segments[segment_index]
        time_kwargs = rs.get_times_kwargs()

        return time_kwargs

    def get_times(self, segment_index=None) -> np.ndarray:
        """Get time vector for a recording segment.

        If the segment has a time_vector, then it is returned. Otherwise
        a time_vector is constructed on the fly with sampling frequency.
        If t_start is defined and the time vector is constructed on the fly,
        the first time will be t_start. Otherwise it will start from 0.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        np.array
            The 1d times array
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        times = rs.get_times()
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
        rs = self._recording_segments[segment_index]
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
        rs = self._recording_segments[segment_index]
        return rs.get_end_time()

    def has_time_vector(self, segment_index=None):
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
        rs = self._recording_segments[segment_index]
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
        rs = self._recording_segments[segment_index]

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
            rs = self._recording_segments[segment_index]
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

        for idx in segments_to_shift:
            rs = self._recording_segments[idx]

            if self.has_time_vector(segment_index=idx):
                rs.time_vector += shift
            else:
                rs.t_start += shift

    def sample_index_to_time(self, sample_ind, segment_index=None):
        """
        Transform sample index into time in seconds
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        return rs.sample_index_to_time(sample_ind)

    def time_to_sample_index(self, time_s, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        rs = self._recording_segments[segment_index]
        return rs.time_to_sample_index(time_s)

    def _get_t_starts(self):
        # handle t_starts
        t_starts = []
        has_time_vectors = []
        for rs in self._recording_segments:
            d = rs.get_times_kwargs()
            t_starts.append(d["t_start"])

        if all(t_start is None for t_start in t_starts):
            t_starts = None
        return t_starts

    def _get_time_vectors(self):
        time_vectors = []
        for rs in self._recording_segments:
            d = rs.get_times_kwargs()
            time_vectors.append(d["time_vector"])
        if all(time_vector is None for time_vector in time_vectors):
            time_vectors = None
        return time_vectors

    def _save(self, format="binary", verbose: bool = False, **save_kwargs):
        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == "binary":
            folder = kwargs["folder"]
            file_paths = [folder / f"traces_cached_seg{i}.raw" for i in range(self.get_num_segments())]
            dtype = kwargs.get("dtype", None) or self.get_dtype()
            t_starts = self._get_t_starts()

            write_binary_recording(self, file_paths=file_paths, dtype=dtype, verbose=verbose, **job_kwargs)

            from .binaryrecordingextractor import BinaryRecordingExtractor

            # This is created so it can be saved as json because the `BinaryFolderRecording` requires it loading
            # See the __init__ of `BinaryFolderRecording`
            binary_rec = BinaryRecordingExtractor(
                file_paths=file_paths,
                sampling_frequency=self.get_sampling_frequency(),
                num_channels=self.get_num_channels(),
                dtype=dtype,
                t_starts=t_starts,
                channel_ids=self.get_channel_ids(),
                time_axis=0,
                file_offset=0,
                is_filtered=self.is_filtered(),
                gain_to_uV=self.get_channel_gains(),
                offset_to_uV=self.get_channel_offsets(),
            )
            binary_rec.dump(folder / "binary.json", relative_to=folder)

            from .binaryfolder import BinaryFolderRecording

            cached = BinaryFolderRecording(folder_path=folder)

        elif format == "memory":
            if kwargs.get("sharedmem", True):
                from .numpyextractors import SharedMemoryRecording

                cached = SharedMemoryRecording.from_recording(self, **job_kwargs)
            else:
                from spikeinterface.core import NumpyRecording

                cached = NumpyRecording.from_recording(self, **job_kwargs)

        elif format == "zarr":
            from .zarrextractors import ZarrRecordingExtractor

            zarr_path = kwargs.pop("zarr_path")
            storage_options = kwargs.pop("storage_options")
            ZarrRecordingExtractor.write_recording(
                self, zarr_path, storage_options, verbose=verbose, **kwargs, **job_kwargs
            )
            cached = ZarrRecordingExtractor(zarr_path, storage_options)

        elif format == "nwb":
            # TODO implement a format based on zarr
            raise NotImplementedError

        else:
            raise ValueError(f"format {format} not supported")

        if self.get_property("contact_vector") is not None:
            probegroup = self.get_probegroup()
            cached.set_probegroup(probegroup)

        for segment_index in range(self.get_num_segments()):
            if self.has_time_vector(segment_index):
                # the use of get_times is preferred since timestamps are converted to array
                time_vector = self.get_times(segment_index=segment_index)
                cached.set_times(time_vector, segment_index=segment_index)

        return cached

    def _extra_metadata_from_folder(self, folder):
        # load probe
        folder = Path(folder)
        if (folder / "probe.json").is_file():
            probegroup = read_probeinterface(folder / "probe.json")
            self.set_probegroup(probegroup, in_place=True)

        # load time vector if any
        for segment_index, rs in enumerate(self._recording_segments):
            time_file = folder / f"times_cached_seg{segment_index}.npy"
            if time_file.is_file():
                time_vector = np.load(time_file)
                rs.time_vector = time_vector

    def _extra_metadata_to_folder(self, folder):
        # save probe
        if self.get_property("contact_vector") is not None:
            probegroup = self.get_probegroup()
            write_probeinterface(folder / "probe.json", probegroup)

        # save time vector if any
        for segment_index, rs in enumerate(self._recording_segments):
            d = rs.get_times_kwargs()
            time_vector = d["time_vector"]
            if time_vector is not None:
                np.save(folder / f"times_cached_seg{segment_index}.npy", time_vector)

    def select_channels(self, channel_ids: list | np.array | tuple) -> "BaseRecording":
        """
        Returns a new recording object with a subset of channels.

        Note that this method does not modify the current recording and instead returns a new recording object.

        Parameters
        ----------
        channel_ids : list or np.array or tuple
            The channel ids to select.
        """
        from .channelslice import ChannelSliceRecording

        return ChannelSliceRecording(self, channel_ids)

    def rename_channels(self, new_channel_ids: list | np.array | tuple) -> "BaseRecording":
        """
        Returns a new recording object with renamed channel ids.

        Note that this method does not modify the current recording and instead returns a new recording object.

        Parameters
        ----------
        new_channel_ids : list or np.array or tuple
            The new channel ids. They are mapped positionally to the old channel ids.
        """
        from .channelslice import ChannelSliceRecording

        assert len(new_channel_ids) == self.get_num_channels(), (
            "new_channel_ids must have the same length as the " "number of channels in the recording"
        )

        return ChannelSliceRecording(self, renamed_channel_ids=new_channel_ids)

    def _channel_slice(self, channel_ids, renamed_channel_ids=None):
        from .channelslice import ChannelSliceRecording

        warnings.warn(
            "Recording.channel_slice will be removed in version 0.103, use `select_channels` or `rename_channels` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        sub_recording = ChannelSliceRecording(self, channel_ids, renamed_channel_ids=renamed_channel_ids)
        return sub_recording

    def _remove_channels(self, remove_channel_ids):
        from .channelslice import ChannelSliceRecording

        recording_channel_ids = self.get_channel_ids()
        non_present_channel_ids = list(set(remove_channel_ids).difference(recording_channel_ids))
        if len(non_present_channel_ids) != 0:
            raise ValueError(
                f"`remove_channel_ids` {non_present_channel_ids} are not in recording ids {recording_channel_ids}."
            )

        new_channel_ids = self.channel_ids[~np.isin(self.channel_ids, remove_channel_ids)]
        sub_recording = ChannelSliceRecording(self, new_channel_ids)
        return sub_recording

    def frame_slice(self, start_frame: int | None, end_frame: int | None) -> BaseRecording:
        """
        Returns a new recording with sliced frames. Note that this operation is not in place.

        Parameters
        ----------
        start_frame : int, optional
            The start frame, if not provided it is set to 0
        end_frame : int, optional
            The end frame, it not provided it is set to the total number of samples

        Returns
        -------
        BaseRecording
            A new recording object with only samples between start_frame and end_frame
        """

        from .frameslicerecording import FrameSliceRecording

        sub_recording = FrameSliceRecording(self, start_frame=start_frame, end_frame=end_frame)
        return sub_recording

    def time_slice(self, start_time: float | None, end_time: float | None) -> BaseRecording:
        """
        Returns a new recording object, restricted to the time interval [start_time, end_time].

        Parameters
        ----------
        start_time : float, optional
            The start time in seconds. If not provided it is set to 0.
        end_time : float, optional
            The end time in seconds. If not provided it is set to the total duration.

        Returns
        -------
        BaseRecording
            A new recording object with only samples between start_time and end_time
        """

        assert self.get_num_segments() == 1, "Time slicing is only supported for single segment recordings."

        start_frame = self.time_to_sample_index(start_time) if start_time else None
        end_frame = self.time_to_sample_index(end_time) if end_time else None

        return self.frame_slice(start_frame=start_frame, end_frame=end_frame)

    def _select_segments(self, segment_indices):
        from .segmentutils import SelectSegmentRecording

        return SelectSegmentRecording(self, segment_indices=segment_indices)

    def get_channel_locations(
        self,
        channel_ids: list | np.ndarray | tuple | None = None,
        axes: "xy" | "yz" | "xz" | "xyz" = "xy",
    ) -> np.ndarray:
        """
        Get the physical locations of specified channels.

        Parameters
        ----------
        channel_ids : array-like, optional
            The IDs of the channels for which to retrieve locations. If None, retrieves locations
            for all available channels. Default is None.
        axes : "xy" | "yz" | "xz" | "xyz", default: "xy"
            The spatial axes to return, specified as a string (e.g., "xy", "xyz"). Default is "xy".

        Returns
        -------
        np.ndarray
            A 2D or 3D array of shape (n_channels, n_dimensions) containing the locations of the channels.
            The number of dimensions depends on the `axes` argument (e.g., 2 for "xy", 3 for "xyz").
        """
        return super().get_channel_locations(channel_ids=channel_ids, axes=axes)

    def is_binary_compatible(self) -> bool:
        """
        Checks if the recording is "binary" compatible.
        To be used before calling `rec.get_binary_description()`

        Returns
        -------
        bool
            True if the underlying recording is binary
        """
        # has to be changed in subclass if yes
        return False

    def get_binary_description(self):
        """
        When `rec.is_binary_compatible()` is True
        this returns a dictionary describing the binary format.
        """
        if not self.is_binary_compatible:
            raise NotImplementedError

    def binary_compatible_with(
        self,
        dtype=None,
        time_axis=None,
        file_paths_length=None,
        file_offset=None,
        file_suffix=None,
        file_paths_lenght=None,
    ):
        """
        Check is the recording is binary compatible with some constrain on

          * dtype
          * tim_axis
          * len(file_paths)
          * file_offset
          * file_suffix
        """

        # spelling typo need to fix
        if file_paths_lenght is not None:
            warnings.warn(
                "`file_paths_lenght` is deprecated and will be removed in 0.103.0 please use `file_paths_length`"
            )
            if file_paths_length is None:
                file_paths_length = file_paths_lenght

        if not self.is_binary_compatible():
            return False

        d = self.get_binary_description()

        if dtype is not None and dtype != d["dtype"]:
            return False

        if time_axis is not None and time_axis != d["time_axis"]:
            return False

        if file_paths_length is not None and file_paths_length != len(d["file_paths"]):
            return False

        if file_offset is not None and file_offset != d["file_offset"]:
            return False

        if file_suffix is not None and not all(Path(e).suffix == file_suffix for e in d["file_paths"]):
            return False

        # good job you pass all crucible
        return True

    def astype(self, dtype, round: bool | None = None):
        from spikeinterface.preprocessing.astype import astype

        return astype(self, dtype=dtype, round=round)


class BaseRecordingSegment(BaseSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

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

    def get_times(self) -> np.ndarray:
        if self.time_vector is not None:
            self.time_vector = np.asarray(self.time_vector)
            return self.time_vector
        else:
            time_vector = np.arange(self.get_num_samples(), dtype="float64")
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
            sample_index = np.round(sample_index).astype(int)
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

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | np.array | tuple | None = None,
    ) -> np.ndarray:
        """
        Return the raw traces, optionally for a subset of samples and/or channels

        Parameters
        ----------
        start_frame : int | None, default: None
            start sample index, or zero if None
        end_frame : int | None, default: None
            end_sample, or number of samples if None
        channel_indices : list | np.array | tuple | None, default: None
            Indices of channels to return, or all channels if None

        Returns
        -------
        traces : np.ndarray
            Array of traces, num_samples x num_channels
        """
        # must be implemented in subclass
        raise NotImplementedError
