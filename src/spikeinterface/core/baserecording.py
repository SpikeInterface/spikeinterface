from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from probeinterface import read_probeinterface, write_probeinterface

from .chunkable import ChunkableSegment, ChunkableMixin
from .baserecordingsnippets import BaseRecordingSnippets
from .core_tools import convert_bytes_to_str, convert_seconds_to_str
from .job_tools import split_job_kwargs


class BaseRecording(BaseRecordingSnippets, ChunkableMixin):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """

    _main_annotations = BaseRecordingSnippets._main_annotations + ["is_filtered"]
    _main_properties = [
        "group",
        "location",
        "gain_to_uV",
        "offset_to_uV",
        "gain_to_physical_unit",
        "offset_to_physical_unit",
        "physical_unit",
    ]
    _main_features = []  # recording do not handle features

    _skip_properties = [
        "noise_level_std_raw",
        "noise_level_std_scaled",
        "noise_level_mad_raw",
        "noise_level_mad_scaled",
        "noise_level_rms_raw",
        "noise_level_rms_scaled",
    ]

    def __init__(self, sampling_frequency: float, channel_ids: list, dtype):
        BaseRecordingSnippets.__init__(
            self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype
        )

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

    def __add__(self, other):
        from .operatorrecordings import AddRecordings

        return AddRecordings(self, other)

    def __sub__(self, other):
        from .operatorrecordings import SubtractRecordings

        return SubtractRecordings(self, other)

    def get_sample_size_in_bytes(self):
        """
        Returns the size of a single sample across all channels in bytes.

        Returns
        -------
        int
            The size of a single sample in bytes
        """
        num_channels = self.get_num_channels()
        dtype_size_bytes = self.get_dtype().itemsize
        sample_size = num_channels * dtype_size_bytes
        return sample_size

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
        return int(self.segments[segment_index].get_num_samples())

    get_num_frames = get_num_samples

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
        return_scaled: bool | None = None,
        return_in_uV: bool = False,
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
        return_scaled : bool | None, default: None
            DEPRECATED. Use return_in_uV instead.
            If True and the recording has scaling (gain_to_uV and offset_to_uV properties),
            traces are scaled to uV
        return_in_uV : bool, default: False
            If True and the recording has scaling (gain_to_uV and offset_to_uV properties),
            traces are scaled to uV

        Returns
        -------
        np.array
            The traces (num_samples, num_channels)

        Raises
        ------
        ValueError
            If return_in_uV is True, but recording does not have scaled traces
        """
        segment_index = self._check_segment_index(segment_index)
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        rs = self.segments[segment_index]
        start_frame = int(start_frame) if start_frame is not None else 0
        num_samples = rs.get_num_samples()
        end_frame = int(min(end_frame, num_samples)) if end_frame is not None else num_samples
        traces = rs.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=channel_indices)
        if order is not None:
            assert order in ["C", "F"]
            traces = np.asanyarray(traces, order=order)

        # Handle deprecated return_scaled parameter
        if return_scaled is not None:
            warnings.warn(
                "`return_scaled` is deprecated and will be removed in version 0.105.0. Use `return_in_uV` instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return_in_uV = return_scaled

        if return_in_uV:
            if not self.has_scaleable_traces():
                if self._dtype.kind == "f":
                    # here we do not truely have scale but we assume this is scaled
                    # this helps a lot for simulated data
                    pass
                else:
                    raise ValueError(
                        "This recording does not support return_in_uV=True (need gain_to_uV and offset_"
                        "to_uV properties)"
                    )
            else:
                gains = self.get_property("gain_to_uV")
                offsets = self.get_property("offset_to_uV")
                gains = gains[channel_indices].astype("float32", copy=False)
                offsets = offsets[channel_indices].astype("float32", copy=False)
                traces = traces.astype("float32", copy=False) * gains + offsets
        return traces

    def get_data(self, start_frame: int, end_frame: int, segment_index: int | None = None, **kwargs) -> np.ndarray:
        """
        General retrieval function for chunkable objects
        """
        return self.get_traces(segment_index=segment_index, start_frame=start_frame, end_frame=end_frame, **kwargs)

    def get_shape(self, segment_index: int | None = None) -> tuple[int, ...]:
        return (self.get_num_samples(segment_index=segment_index), self.get_num_channels())

    def _save(self, format="binary", verbose: bool = False, **save_kwargs):
        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == "binary":
            from .chunkable_tools import write_binary

            folder = kwargs["folder"]
            file_paths = [folder / f"traces_cached_seg{i}.raw" for i in range(self.get_num_segments())]
            dtype = kwargs.get("dtype", None) or self.get_dtype()
            t_starts = self._get_t_starts()

            write_binary(self, file_paths=file_paths, dtype=dtype, verbose=verbose, **job_kwargs)

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
        for segment_index, rs in enumerate(self.segments):
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
        for segment_index, rs in enumerate(self.segments):
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
            Start frame index. If None, defaults to the beginning of the recording (frame 0).
        end_frame : int, optional
            End frame index. If None, defaults to the last frame of the recording.

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
            Start time in seconds. If None, defaults to the beginning of the recording.
        end_time : float, optional
            End time in seconds. If None, defaults to the end of the recording.

        Returns
        -------
        BaseRecording
            A new recording object with only samples between start_time and end_time
        """
        num_segments = self.get_num_segments()
        assert (
            num_segments == 1
        ), f"Time slicing is only supported for single segment recordings. Found {num_segments} segments."

        t_start = self.get_start_time()
        t_end = self.get_end_time()

        if start_time is not None:
            t_start = self.get_start_time()
            t_start_too_early = start_time < t_start
            if t_start_too_early:
                raise ValueError(f"start_time {start_time} is before the start time {t_start} of the recording.")
            t_start_too_late = start_time > t_end
            if t_start_too_late:
                raise ValueError(f"start_time {start_time} is after the end time {t_end} of the recording.")
            start_frame = self.time_to_sample_index(start_time)
        else:
            start_frame = None

        if end_time is not None:
            t_end_too_early = end_time < t_start
            if t_end_too_early:
                raise ValueError(f"end_time {end_time} is before the start time {t_start} of the recording.")

            t_end_too_late = end_time > t_end
            if t_end_too_late:
                raise ValueError(f"end_time {end_time} is after the end time {t_end} of the recording.")
            end_frame = self.time_to_sample_index(end_time)
        else:
            end_frame = None

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
    ):
        """
        Check is the recording is binary compatible with some constrain on

          * dtype
          * tim_axis
          * len(file_paths)
          * file_offset
          * file_suffix
        """

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


class BaseRecordingSegment(ChunkableSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

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
