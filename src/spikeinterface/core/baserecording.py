import warnings
from typing import Literal
from pathlib import Path

import numpy as np
from probeinterface import Probe, ProbeGroup, select_axes

from .time_series import TimeSeriesSegment, TimeSeries
from .base import BaseExtractor
from .core_tools import convert_bytes_to_str, convert_seconds_to_str
from .job_tools import split_job_kwargs
from .recording_tools import _set_group_property_based_on_probegroup, check_probe_do_not_overlap


class BaseRecording(BaseExtractor, TimeSeries):
    """
    Abstract class representing several a multichannel timeseries (or block of raw ephys traces).
    Internally handle list of RecordingSegment
    """

    _main_annotations = BaseExtractor._main_annotations + ["is_filtered"]
    _main_properties = [
        "group",
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
        BaseExtractor.__init__(self, channel_ids)
        TimeSeries.__init__(self)
        self._sampling_frequency = float(sampling_frequency)
        self._dtype = np.dtype(dtype)
        self._probegroup = None
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

    @property
    def channel_ids(self):
        return self._main_ids

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def dtype(self):
        return self._dtype

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_ids(self):
        return self._main_ids

    def get_num_channels(self):
        return len(self.get_channel_ids())

    def get_dtype(self):
        return self._dtype

    # TimeSeries zone
    @property
    def segments(self) -> list["BaseRecordingSegment"]:
        """List of recording segments."""
        return self._segments

    @property
    def _recording_segments(self) -> list["BaseRecordingSegment"]:
        """For backward compatibility, we keep _recording_segments."""
        return self._segments

    def add_recording_segment(self, recording_segment: "BaseRecordingSegment") -> None:
        """Adds a recording segment.

        Parameters
        ----------
        recording_segment : BaseRecordingSegment
            The recording segment to add
        """
        super().add_segment(recording_segment)

    def get_sample_size_in_bytes(self, dtype=None):
        """
        Returns the size of a single sample across all channels in bytes.

        Parameters
        ----------
        dtype : data-type, optional
            The data type to use for calculating the sample size. If None,
            the recording's dtype is used.

        Returns
        -------
        int
            The size of a single sample in bytes
        """
        num_channels = self.get_num_channels()
        dtype = self.get_dtype() if dtype is None else np.dtype(dtype)
        dtype_size_bytes = dtype.itemsize
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

    def get_traces(
        self,
        segment_index: int | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_ids: list | np.ndarray | tuple | None = None,
        order: Literal["C", "F"] | None = None,
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
        channel_ids : list | np.ndarray | tuple | None, default: None
            The channel ids. If None, all channels are used, default: None
        order : "C" | "F" | None, default: None
            The order of the traces ("C" | "F"). If None, traces are returned as they are
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

        if return_in_uV:
            if not self.has_scaleable_traces():
                if self._dtype.kind == "f":
                    # here we do not truly have scale but we assume this is scaled
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
        General retrieval function for time_series objects
        """
        return self.get_traces(segment_index=segment_index, start_frame=start_frame, end_frame=end_frame, **kwargs)

    def get_shape(self, segment_index: int | None = None) -> tuple[int, ...]:
        return (self.get_num_samples(segment_index=segment_index), self.get_num_channels())

    # Main properties
    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get("is_filtered", False)

    def set_channel_groups(self, groups, channel_ids=None):
        if "probes" in self._annotations:
            warnings.warn("set_channel_groups() destroys the probe description. Using set_probe() is preferable")
            self._annotations.pop("probes")
        self.set_property("group", groups, ids=channel_ids)

    def get_channel_groups(self, channel_ids=None):
        groups = self.get_property("group", ids=channel_ids)
        return groups

    def clear_channel_groups(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channels()
        else:
            n = len(channel_ids)
        groups = np.zeros(n, dtype="int64")
        self.set_property("group", groups, ids=channel_ids)

    def set_channel_gains(self, gains, channel_ids=None):
        if np.isscalar(gains):
            gains = [gains] * self.get_num_channels()
        self.set_property("gain_to_uV", gains, ids=channel_ids)

    def get_channel_gains(self, channel_ids=None):
        return self.get_property("gain_to_uV", ids=channel_ids)

    def set_channel_offsets(self, offsets, channel_ids=None):
        if np.isscalar(offsets):
            offsets = [offsets] * self.get_num_channels()
        self.set_property("offset_to_uV", offsets, ids=channel_ids)

    def get_channel_offsets(self, channel_ids=None):
        return self.get_property("offset_to_uV", ids=channel_ids)

    def get_channel_property(self, channel_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(channel_id)]
        return v

    # Probe
    def has_scaleable_traces(self) -> bool:
        if self.get_property("gain_to_uV") is None or self.get_property("offset_to_uV") is None:
            return False
        else:
            return True

    def has_probe(self) -> bool:
        # probe group is saved and loaded to binary/zarr, so we don't need to check for legacy "contact_vector" property
        return self._probegroup is not None

    def has_3d_probe(self) -> bool:
        if self.has_probe():
            probe = self.get_probegroup().probes[0]
            return probe.ndim == 3
        else:
            return False

    def has_channel_location(self) -> bool:
        return self.has_probe()

    def remove_probe(self):
        """
        Removes probe information
        """
        self._probegroup = None

    def set_probe(
        self,
        probe: Probe,
        group_mode: Literal["auto", "by_probe", "by_shank", "by_side"] = "auto",
        in_place: bool | None = None,
    ) -> None:
        """
        Attach a Probe object to a recording.

        Parameters
        ----------
        probe: Probe
            The probe to be attached to the recording
        group_mode: "auto" | "by_probe" | "by_shank" | "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks
            and two sides are present.
        in_place: (deprecated) bool | None, default: None
            Deprecated argument to indicate whether to modify the recording in place
            or return a new recording. The function is always in place now.
            Use the `recording.select_channels_with_probegroup()` method instead of `in_place=False`
            to return a new recording with a channel selection to match the probe/probegroup.

        Notes
        -----
        Internally, this will construct a ProbeGroup with the probe and call `set_probegroup()`.
        """
        assert isinstance(probe, Probe), "The input must be a Probe object"
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        # TODO: remove return in 0.106.0 after removing in_place argument
        return self.set_probegroup(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probegroup(
        self,
        probegroup: ProbeGroup,
        group_mode: Literal["auto", "by_probe", "by_shank", "by_side"] = "auto",
        in_place: bool | None = None,
        check_overlap: bool = True,
    ) -> None:
        """
        Attach a ProbeGroup or dict to a recording.
        For this Probe.device_channel_indices is used to link contacts to recording channels.
        After removing unconnected contacts, the number of connected contacts must match the
        number of channels in the recording. If this is not the case, use the `recording.select_with_probegroup()`
        method instead to return a new recording with a channel selection to match the probe/probegroup.

        Note: The probe order of the probegroup is not kept. Channel ids are re-ordered to match the channel_ids of the recording.

        Parameters
        ----------
        probe_or_probegroup: ProbeGroup, or dict
            The probe(s) to be attached to the recording
        group_mode: "auto" | "by_probe" | "by_shank" | "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks and two sides are present.
        in_place: (deprecated) bool | None, default: None
            Deprecated argument to indicate whether to modify the recording in place
            or return a new recording. The function is always in place now.
            Use the `recording.select_channels_with_probegroup()` method instead of `in_place=False`
            to return a new recording with a channel selection to match the probe/probegroup.
        check_overlap: bool, default: True
            If True, check that the probes in the probegroup do not overlap in space.
            This should be set to False when aggregating recordings whose probes share
            the same physical space (e.g. channels split by group from a single probe),
            where contact positions are unique but probe bounding boxes may overlap.
        """
        if in_place is not None:
            warnings.warn(
                "The 'in_place' argument is deprecated and will be removed in version 0.106.0. "
                "The `set_probe/probegroup()` are now always in place; please remove the in_place argument.",
                FutureWarning,
                stacklevel=2,
            )
            if not in_place:
                return self.select_channels_with_probegroup(probegroup, group_mode=group_mode)

        if check_overlap and len(probegroup.probes) > 0:
            check_probe_do_not_overlap(probegroup.probes)

        probegroup_sorted = self._get_probegroup_based_on_device_channel_indices(probegroup)

        if probegroup_sorted.get_contact_count() != self.get_num_channels():
            raise ValueError(
                "The probe/probegroup must have the same number of connected contacts "
                f"as the number of channels as the recording, but the probe has {probegroup.get_contact_count()} "
                f"connected channels and the recording has {self.get_num_channels()} channels. "
                "Use the `recording.select_channels_with_probegroup()` method instead to return a new recording with "
                "a channel selection to match the probe/probegroup."
            )

        device_channel_indices = probegroup_sorted.get_global_device_channel_indices()["device_channel_indices"]
        if not np.array_equal(device_channel_indices, np.arange(self.get_num_channels())):
            raise ValueError(
                "`device_channel_indices` is wrong! "
                "It should contain only values [0...n-1] after ordering, "
                f"but they are: {device_channel_indices}"
            )

        # probegroup_sorted.set_global_device_channel_indices(np.arange(probegroup_sorted.get_contact_count()))
        self._probegroup = probegroup_sorted

        # Handle and set channel groups
        _set_group_property_based_on_probegroup(self, probegroup_sorted, group_mode=group_mode)

    def select_channels_with_probe(
        self, probe: Probe, group_mode: Literal["auto", "by_probe", "by_shank", "by_side"] = "auto"
    ) -> "BaseRecording":
        """
        Returns a new recording with channels selected based on the probe.

        Parameters
        ----------
        probe: Probe
            The probe to be used for channel selection
        group_mode: "auto" | "by_probe" | "by_shank" |
            "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks and two sides are present.

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        assert isinstance(probe, Probe), "The input must be a Probe object"
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self.select_channels_with_probegroup(probegroup, group_mode=group_mode)

    def select_channels_with_probegroup(
        self, probegroup: ProbeGroup, group_mode: Literal["auto", "by_probe", "by_shank", "by_side"] = "auto"
    ) -> "BaseRecording":
        """
        Selects channels based on the given ProbeGroup and returns a new recording with the selected channels.

        Parameters
        ----------
        probegroup: ProbeGroup
            The probegroup to be used for channel selection
        group_mode: "auto" | "by_probe" | "by_shank" |
            "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks
            and two sides are present.

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        probegroup_sorted = self._get_probegroup_based_on_device_channel_indices(probegroup)
        if probegroup_sorted.get_contact_count() > 0:
            sorted_dci = probegroup_sorted.get_global_device_channel_indices()["device_channel_indices"]
            new_channel_ids = self.channel_ids[sorted_dci]
            probegroup_sorted.set_global_device_channel_indices(np.arange(len(new_channel_ids)))
            if np.array_equal(new_channel_ids, self.channel_ids):
                sub_recording = self.clone()
            else:
                sub_recording = self.select_channels(new_channel_ids)
            sub_recording._probegroup = probegroup_sorted
            _set_group_property_based_on_probegroup(sub_recording, probegroup_sorted, group_mode=group_mode)
        else:
            sub_recording = self.select_channels([])  # empty recording
            sub_recording._probegroup = ProbeGroup()  # empty probegroup
        return sub_recording

    def _get_probegroup_based_on_device_channel_indices(self, probegroup: ProbeGroup) -> ProbeGroup:
        """
        Returns a new probegroup sorted based on their device_channel_indices.
        This is useful to ensure that the probes are ordered correctly when attached to a recording.
        Also checks that the device_channel_indices are consistent with the recording channel count and
        contacts are unique across probes in the probegroup.

        Parameters
        ----------
        probegroup : ProbeGroup
            The probegroup to be sorted.

        Returns
        -------
        ProbeGroup
            The sorted probegroup.
        """
        if not isinstance(probegroup, ProbeGroup):
            raise ValueError("The input must be a ProbeGroup or dict")

        assert all(
            probe.device_channel_indices is not None for probe in probegroup.probes
        ), "Probe must have device_channel_indices"

        # Remove unconnected contacts and slice the probe group accordingly
        device_channel_indices = probegroup.get_global_device_channel_indices()["device_channel_indices"]
        keep_indices = np.flatnonzero(device_channel_indices >= 0)
        if len(keep_indices) < len(device_channel_indices):
            if len(keep_indices) == 0:
                device_channel_indices = np.array([], dtype="int64")
            else:
                probegroup = probegroup.get_slice(keep_indices)
                device_channel_indices = device_channel_indices[keep_indices]

        if len(device_channel_indices) > 0:
            # Check consistency of device_channel_indices with the recording channel count
            number_of_device_channel_indices = np.max(list(device_channel_indices) + [0])
            if number_of_device_channel_indices >= self.get_num_channels():
                error_msg = (
                    f"The given Probe either has 'device_channel_indices' that does not match channel count \n"
                    f"{len(device_channel_indices)} vs {self.get_num_channels()} \n"
                    f"or it's max index {number_of_device_channel_indices} is the same as the number of channels {self.get_num_channels()} \n"
                    f"If using all channels remember that python is 0-indexed so max device_channel_index should be {self.get_num_channels() - 1} \n"
                    f"device_channel_indices are the following: {device_channel_indices} \n"
                    f"recording channels are the following: {self.get_channel_ids()} \n"
                )
                raise ValueError(error_msg)
            # Now slice the probe using the device channel indices to match the recording channel_ids
            order = np.argsort(device_channel_indices)
            probegroup = probegroup.get_slice(order)
        else:
            warnings.warn(
                "No connected channels in the probegroup! "
                "The probegroup will be attached but no channel will be selected."
            )
            probegroup = ProbeGroup()  # empty probegroup

        return probegroup

    def get_probe(self):
        probes = self.get_probes()
        assert len(probes) == 1, "There are several probe use .get_probes() or get_probegroup()"
        return probes[0]

    def get_probes(self):
        probegroup = self.get_probegroup()
        return probegroup.probes

    def get_probegroup(self):
        if self._probegroup is None:
            raise ValueError("There is no Probe attached to this recording. Use set_probe(...) to attach one.")
        return self._probegroup

    def create_dummy_probe_from_locations(self, locations, shape="circle", shape_params={"radius": 1}, axes="xy"):
        """
        Creates a "dummy" probe based on locations.

        Parameters
        ----------
        locations : np.array
            Array with channel locations (num_channels, ndim) [ndim can be 2 or 3]
        shape : str, default: "circle"
            Electrode shapes
        shape_params : dict, default: {"radius": 1}
            Shape parameters
        axes : str, default: "xy"
            If ndim is 3, indicates the axes that define the plane of the electrodes

        Returns
        -------
        probe : Probe
            The created probe
        """
        ndim = locations.shape[1]
        probe = Probe(ndim=2)
        if ndim == 3:
            locations_2d = select_axes(locations, axes)
        else:
            locations_2d = locations
        probe.set_contacts(locations_2d, shapes=shape, shape_params=shape_params)
        probe.set_device_channel_indices(np.arange(self.get_num_channels()))

        if ndim == 3:
            probe = probe.to_3d(axes=axes)

        return probe

    def set_dummy_probe_from_locations(self, locations, shape="circle", shape_params={"radius": 1}, axes="xy"):
        """
        Sets a "dummy" probe based on locations.

        Parameters
        ----------
        locations : np.array
            Array with channel locations (num_channels, ndim) [ndim can be 2 or 3]
        shape : str, default: "circle"
            Electrode shapes
        shape_params : dict, default: {"radius": 1}
            Shape parameters
        axes : "xy" | "yz" | "xz", default: "xy"
            If ndim is 3, indicates the axes that define the plane of the electrodes
        """
        probe = self.create_dummy_probe_from_locations(
            np.array(locations), shape=shape, shape_params=shape_params, axes=axes
        )
        self.set_probe(probe)

    def set_channel_locations(self, locations, channel_ids=None):
        warnings.warn(
            (
                "set_channel_locations() is deprecated and will be removed in version 0.106.0. "
                "If you want to set probe information, use `set_dummy_probe_from_locations()`."
            ),
            FutureWarning,
            stacklevel=2,
        )
        self.set_dummy_probe_from_locations(locations, axes="xy")

    def get_channel_locations(
        self,
        channel_ids: list | np.ndarray | tuple | None = None,
        axes: Literal["xy", "yz", "xz", "xyz"] = "xy",
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
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        channel_indices = self.ids_to_indices(channel_ids)
        if not self.has_probe():
            raise ValueError("get_channel_locations(..) needs a probe to be attached to the recording")
        probegroup = self.get_probegroup()
        contact_positions = probegroup.get_global_contact_positions()
        return select_axes(contact_positions, axes)[channel_indices]

    def is_probe_3d(self) -> bool:
        if not self.has_probe():
            raise ValueError("is_probe_3d() needs a probe to be attached to the recording")
        probegroup = self.get_probegroup()
        return probegroup.ndim == 3

    def clear_channel_locations(self, channel_ids=None):
        warnings.warn(
            (
                "clear_channel_locations() is deprecated and will be removed in version 0.106.0. "
                "If you want to remove probe information, use `remove_probe()`."
            ),
            FutureWarning,
            stacklevel=2,
        )
        self.remove_probe()

    def planarize(self, axes: str = "xy"):
        """
        Returns a Recording with a 2D probe from one with a 3D probe

        Parameters
        ----------
        axes : "xy" | "yz" |"xz", default: "xy"
            The axes to keep

        Returns
        -------
        BaseRecording
            The recording with 2D positions
        """
        assert self.has_3d_probe(), "The 'planarize' function needs a recording with 3d locations"
        assert len(axes) == 2, "You need to specify 2 dimensions (e.g. 'xy', 'zy')"

        probe2d = self.get_probe().to_2d(axes=axes)
        recording2d = self.clone()
        recording2d.set_probe(probe2d)

        return recording2d

    def split_by(self, property="group", outputs="dict"):
        """
        Splits object based on a certain property (e.g. "group")

        Parameters
        ----------
        property : str, default: "group"
            The property to use to split the object, default: "group"
        outputs : "dict" | "list", default: "dict"
            Whether to return a dict or a list

        Returns
        -------
        dict or list
            A dict or list with grouped objects based on property

        Raises
        ------
        ValueError
            Raised when property is not present
        """
        assert outputs in ("list", "dict")
        values = self.get_property(property)
        if values is None:
            raise ValueError(f"property {property} is not set")

        if outputs == "list":
            recordings = []
        elif outputs == "dict":
            recordings = {}
        for value in np.unique(values).tolist():
            (inds,) = np.nonzero(values == value)
            new_channel_ids = self.channel_ids[inds]
            subrec = self.select_channels(new_channel_ids)
            subrec.set_annotation("split_by_property", value=property)
            if outputs == "list":
                recordings.append(subrec)
            elif outputs == "dict":
                recordings[value] = subrec
        return recordings

    # Save and metadata handling and propagation
    def save(self, format="binary", verbose: bool = False, **save_kwargs):
        """
        Save a `BaseRecording` object to a specified format:

        * "binary"
        * "zarr"
        * "memory"

        Parameters
        ----------
        format : str, default: "binary"
            The format to save the recording in. Options are:

            - "binary": Saves the recording in binary format.
            - "zarr": Saves the recording in Zarr format.
            - "memory": Saves the recording in memory (shared memory or numpy array).
        verbose : bool, default: False
            If True, prints additional information during the save process.
        **save_kwargs : dict
            Additional keyword arguments specific to the chosen format.
            All formats support job_kwargs for parallel processing
            (see `si.get_global_job_kwargs()` for default values).

            * "binary" format:
                - folder : str or Path
                    The folder where the binary files will be saved.
                - overwrite : bool, default: False
                    If True, existing files in the folder will be overwritten.
                - dtype : str, optional
                    The data type to use for saving the recording. If not provided, the recording's dtype
                    will be used.
            * "zarr" format:
                - folder : str or Path
                    The folder where the Zarr files will be saved.
                - overwrite: bool, default: False
                    If True, the folder is removed if it already exists
                - storage_options: dict or None, default: None
                    Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.
                    For cloud storage locations, this should not be None (in case of default values, use an empty dict)
                - channel_chunk_size: int or None, default: None
                    Channels per chunk (only for BaseRecording)
                - compressor: numcodecs.Codec or None, default: None
                    Global compressor. If None, Blosc-zstd, level 5, with bit shuffle is used
                - filters: list[numcodecs.Codec] or None, default: None
                    Global filters for zarr (global)
                - compressor_by_dataset: dict or None, default: None
                    Optional compressor per dataset:

                        - traces
                        - times

                    If None, the global compressor is used
                - filters_by_dataset: dict or None, default: None
                    Optional filters per dataset:

                        - traces
                        - times

                    If None, the global filters are used
            * "memory" format:
                - sharedmem : bool, default: True
                    If True, the recording is saved in shared memory. If False, it is saved as
                    a numpy array in memory.

        Returns
        -------
        BaseRecording
            The saved recording object in the specified format.
        """
        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == "binary":
            if "folder" not in kwargs:
                raise ValueError("Missing folder in recording.save(folder='...')")

            from .binaryfolder import BinaryFolderRecording

            folder = kwargs.pop("folder")
            cached = BinaryFolderRecording.write_recording(
                self, folder_path=folder, verbose=verbose, **kwargs, **job_kwargs
            )

        elif format == "memory":
            if kwargs.get("sharedmem", True):
                from .numpyextractors import SharedMemoryRecording

                cached = SharedMemoryRecording.from_recording(
                    self, with_metadata=True, with_time_vector=True, **job_kwargs
                )
            else:
                from spikeinterface.core import NumpyRecording

                cached = NumpyRecording.from_recording(self, with_metadata=True, with_time_vector=True, **job_kwargs)

        elif format == "zarr":
            if "folder" not in kwargs:
                raise ValueError("Missing folder in recording.save(folder='...')")
            folder_path = kwargs.pop("folder")

            from .zarrextractors import ZarrRecordingExtractor

            cached = ZarrRecordingExtractor.write_recording(
                self, folder_path=folder_path, verbose=verbose, **kwargs, **job_kwargs
            )

        else:
            raise ValueError(f"format {format} not supported")

        return cached

    def _extra_metadata_to_dict(self, dump_dict):
        # Save probe
        if self.has_probe():
            probegroup = self.get_probegroup()
            dump_dict["probegroup"] = probegroup.to_dict()

        # Add times_kwargs if the recording has been modified in memory (e.g. by set_times / shift_times / reset_times)
        if self._time_info_modified:
            dump_dict["times_kwargs"] = []
            for segment_index in range(self.get_num_segments()):
                times_kwargs = self.segments[segment_index].get_times_kwargs()
                dump_dict["times_kwargs"].append(times_kwargs)

    def _extra_metadata_from_dict(self, dump_dict):
        # Load probe and handle backward-compatibility with legacy "contact_vector"/"location" property
        if "probegroup" in dump_dict:
            # this is for SI>=0.105.0
            probegroup = dump_dict["probegroup"]
            self._probegroup = ProbeGroup.from_dict(probegroup)

        if "times_kwargs" in dump_dict:
            # When serializing, dump timestamps information because this could have been
            # set in memory
            times_kwargs_list = dump_dict["times_kwargs"]
            for segment_index, times_kwargs in enumerate(times_kwargs_list):
                self.segments[segment_index]._sampling_frequency = times_kwargs["sampling_frequency"]
                self.segments[segment_index]._t_start = times_kwargs["t_start"]
                self.segments[segment_index]._time_vector = times_kwargs["time_vector"]

    def _extra_metadata_copy(self, other):
        if self._probegroup is not None:
            other._probegroup = self._probegroup.copy()

    def _handle_extractor_backward_compatibility(self):
        """
        This handles backward compatibility for recordings that were saved with older versions of spikeinterface.

        Options:

        1. "contact_vector" property: This was used in versions < 0.105.0 to store probe information, when saved to
            pickle
        2. "location" property: This was used in versions < 0.105.0 to store probe information, when saved to JSON
            (no contact_vector saved)
        3. probe annotation: probe annotations and contours were saved as recording properties in versions < 0.105.0,
            but now they are saved in the probegroup. This method will copy the annotations and the contour to the probes
            in the the probegroup and remove the annotations from the recording.
        """
        if self._probegroup is None:
            check_for_probes_info = False
            if "contact_vector" in self.get_property_keys():
                # this is for SI<0.105.0 and from pickle
                contact_vector = self.get_property("contact_vector")
                probegroup = ProbeGroup.from_numpy(contact_vector)
                self._probegroup = probegroup
                check_for_probes_info = True
            elif "location" in self.get_property_keys():
                # this is for SI<0.105.0 and from JSON (no contact_vector saved)
                locations = self.get_property("location")
                self.set_dummy_probe_from_locations(locations)
                check_for_probes_info = True

            if check_for_probes_info:
                for i, probe in enumerate(self._probegroup.probes):
                    if "probes_info" in self._annotations:
                        probe_dict = self._annotations["probes_info"][i]
                        probe.annotations.update(probe_dict)
                    if f"probe_{i}_planar_contour" in self._annotations:
                        contour = self.get_annotation(f"probe_{i}_planar_contour")
                        if contour is not None:
                            probe.set_planar_contour(contour)
                        self.delete_annotation(f"probe_{i}_planar_contour")
        if "probes_info" in self._annotations:
            self._annotations.pop("probes_info")

    # Utility methods for channel/time manipulation
    def select_channels(self, channel_ids: list | np.ndarray | tuple) -> "BaseRecording":
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

    def remove_channels(self, remove_channel_ids):
        """
        Returns a new object with removed channels.


        Parameters
        ----------
        remove_channel_ids : np.array or list
            The list of channels to remove

        Returns
        -------
        BaseRecording
            The object with removed channels
        """
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

    def select_segments(self, segment_indices):
        """
        Return a new object with the segments specified by "segment_indices".

        Parameters
        ----------
        segment_indices : list of int
            List of segment indices to keep in the returned recording

        Returns
        -------
        BaseRecording
            The onject with the selected segments
        """
        from .segmentutils import SelectSegmentRecording

        return SelectSegmentRecording(self, segment_indices=segment_indices)

    def rename_channels(self, new_channel_ids: list | np.ndarray | tuple) -> "BaseRecording":
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

    def frame_slice(self, start_frame: int | None, end_frame: int | None) -> "BaseRecording":
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

    def time_slice(self, start_time: float | None, end_time: float | None) -> "BaseRecording":
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

    def astype(self, dtype, round: bool | None = None):
        from spikeinterface.preprocessing.astype import astype

        return astype(self, dtype=dtype, round=round)

    # Binary compatibility
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


class BaseRecordingSegment(TimeSeriesSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    # Segments that know their channel count at construction (e.g. BinaryRecordingSegment,
    # which needs it before being attached to a parent to compute the on-disk layout) set
    # self.num_channels. Segments that don't leave it unset and inherit the count from the
    # parent recording, which is always attached by the time get_traces runs.
    def get_num_channels(self) -> int:
        if hasattr(self, "num_channels") and self.num_channels is not None:
            return self.num_channels
        return self.parent_extractor.get_num_channels()

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | np.ndarray | tuple | None = None,
    ) -> np.ndarray:
        """
        Return the raw traces, optionally for a subset of samples and/or channels

        Parameters
        ----------
        start_frame : int | None, default: None
            start sample index, or zero if None
        end_frame : int | None, default: None
            end_sample, or number of samples if None
        channel_indices : list | np.ndarray | tuple | None, default: None
            Indices of channels to return, or all channels if None

        Returns
        -------
        traces : np.ndarray
            Array of traces, num_samples x num_channels
        """
        # must be implemented in subclass
        raise NotImplementedError

    def get_data(
        self, start_frame: int, end_frame: int, indices: list | np.ndarray | tuple | None = None
    ) -> np.ndarray:
        """
        General retrieval function for time_series objects
        """
        return self.get_traces(start_frame=start_frame, end_frame=end_frame, channel_indices=indices)
