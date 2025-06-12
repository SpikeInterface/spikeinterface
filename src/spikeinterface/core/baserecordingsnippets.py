from __future__ import annotations
from pathlib import Path

import numpy as np

from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface, select_axes

from .base import BaseExtractor
from .recording_tools import check_probe_do_not_overlap

from warnings import warn


class BaseRecordingSnippets(BaseExtractor):
    """
    Mixin that handles all probe and channel operations
    """

    def __init__(self, sampling_frequency: float, channel_ids: list[str, int], dtype: np.dtype):
        BaseExtractor.__init__(self, channel_ids)
        self._sampling_frequency = float(sampling_frequency)
        self._dtype = np.dtype(dtype)

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

    def has_scaleable_traces(self) -> bool:
        if self.get_property("gain_to_uV") is None or self.get_property("offset_to_uV") is None:
            return False
        else:
            return True

    def has_scaled(self):
        warn(
            "`has_scaled` has been deprecated and will be removed in 0.103.0. Please use `has_scaleable_traces()`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.has_scaleable_traces()

    def has_probe(self) -> bool:
        return "contact_vector" in self.get_property_keys()

    def has_channel_location(self) -> bool:
        return self.has_probe() or "location" in self.get_property_keys()

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get("is_filtered", False)

    def _channel_slice(self, channel_ids, renamed_channel_ids=None):
        raise NotImplementedError

    def set_probe(self, probe, group_mode="by_probe", in_place=False):
        """
        Attach a list of Probe object to a recording.

        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording
        group_mode: "by_probe" | "by_shank", default: "by_probe
            "by_probe" or "by_shank". Adds grouping property to the recording based on the probes ("by_probe")
            or  shanks ("by_shanks")
        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        assert isinstance(probe, Probe), "must give Probe"
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self._set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probegroup(self, probegroup, group_mode="by_probe", in_place=False):
        return self._set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def _set_probes(self, probe_or_probegroup, group_mode="by_probe", in_place=False):
        """
        Attach a list of Probe objects to a recording.
        For this Probe.device_channel_indices is used to link contacts to recording channels.
        If some contacts of the Probe are not connected (device_channel_indices=-1)
        then the recording is "sliced" and only connected channel are kept.

        The probe order is not kept. Channel ids are re-ordered to match the channel_ids of the recording.


        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording
        group_mode: "by_probe" | "by_shank", default: "by_probe"
            "by_probe" or "by_shank". Adds grouping property to the recording based on the probes ("by_probe")
            or  shanks ("by_shank")
        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        assert group_mode in ("by_probe", "by_shank"), "'group_mode' can be 'by_probe' or 'by_shank'"

        # handle several input possibilities
        if isinstance(probe_or_probegroup, Probe):
            probegroup = ProbeGroup()
            probegroup.add_probe(probe_or_probegroup)
        elif isinstance(probe_or_probegroup, ProbeGroup):
            probegroup = probe_or_probegroup
        elif isinstance(probe_or_probegroup, list):
            assert all([isinstance(e, Probe) for e in probe_or_probegroup])
            probegroup = ProbeGroup()
            for probe in probe_or_probegroup:
                probegroup.add_probe(probe)
        else:
            raise ValueError("must give Probe or ProbeGroup or list of Probe")

        # check that the probe do not overlap
        num_probes = len(probegroup.probes)
        if num_probes > 1:
            check_probe_do_not_overlap(probegroup.probes)

        # handle not connected channels
        assert all(
            probe.device_channel_indices is not None for probe in probegroup.probes
        ), "Probe must have device_channel_indices"

        # this is a vector with complex fileds (dataframe like) that handle all contact attr
        probe_as_numpy_array = probegroup.to_numpy(complete=True)

        # keep only connected contact ( != -1)
        keep = probe_as_numpy_array["device_channel_indices"] >= 0
        if np.any(~keep):
            warn("The given probes have unconnected contacts: they are removed")

        probe_as_numpy_array = probe_as_numpy_array[keep]
        device_channel_indices = probe_as_numpy_array["device_channel_indices"]
        order = np.argsort(device_channel_indices)
        device_channel_indices = device_channel_indices[order]

        # check TODO: Where did this came from?
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

        new_channel_ids = self.get_channel_ids()[device_channel_indices]
        probe_as_numpy_array = probe_as_numpy_array[order]
        probe_as_numpy_array["device_channel_indices"] = np.arange(probe_as_numpy_array.size, dtype="int64")

        # create recording : channel slice or clone or self
        if in_place:
            if not np.array_equal(new_channel_ids, self.get_channel_ids()):
                raise Exception("set_probe(inplace=True) must have all channel indices")
            sub_recording = self
        else:
            if np.array_equal(new_channel_ids, self.get_channel_ids()):
                sub_recording = self.clone()
            else:
                sub_recording = self.select_channels(new_channel_ids)

        # create a vector that handle all contacts in property
        sub_recording.set_property("contact_vector", probe_as_numpy_array, ids=None)

        # planar_contour is saved in annotations
        for probe_index, probe in enumerate(probegroup.probes):
            contour = probe.probe_planar_contour
            if contour is not None:
                sub_recording.set_annotation(f"probe_{probe_index}_planar_contour", contour, overwrite=True)

        # duplicate positions to "locations" property
        ndim = probegroup.ndim
        locations = np.zeros((probe_as_numpy_array.size, ndim), dtype="float64")
        for i, dim in enumerate(["x", "y", "z"][:ndim]):
            locations[:, i] = probe_as_numpy_array[dim]
        sub_recording.set_property("location", locations, ids=None)

        # handle groups
        groups = np.zeros(probe_as_numpy_array.size, dtype="int64")
        if group_mode == "by_probe":
            for group, probe_index in enumerate(np.unique(probe_as_numpy_array["probe_index"])):
                mask = probe_as_numpy_array["probe_index"] == probe_index
                groups[mask] = group
        elif group_mode == "by_shank":
            assert all(
                probe.shank_ids is not None for probe in probegroup.probes
            ), "shank_ids is None in probe, you cannot group by shank"
            for group, a in enumerate(np.unique(probe_as_numpy_array[["probe_index", "shank_ids"]])):
                mask = (probe_as_numpy_array["probe_index"] == a["probe_index"]) & (
                    probe_as_numpy_array["shank_ids"] == a["shank_ids"]
                )
                groups[mask] = group
        sub_recording.set_property("group", groups, ids=None)

        # add probe annotations to recording
        probes_info = []
        for probe in probegroup.probes:
            probes_info.append(probe.annotations)
        self.annotate(probes_info=probes_info)

        return sub_recording

    def set_probes(self, probe_or_probegroup, group_mode="by_probe", in_place=False):

        warning_msg = (
            "`set_probes` is now a private function and the public function will be "
            "removed in 0.103.0. Please use `set_probe` or `set_probegroup` instead"
        )

        warn(warning_msg, category=DeprecationWarning, stacklevel=2)

        sub_recording = self._set_probes(
            probe_or_probegroup=probe_or_probegroup, group_mode=group_mode, in_place=in_place
        )

        return sub_recording

    def get_probe(self):
        probes = self.get_probes()
        assert len(probes) == 1, "there are several probe use .get_probes() or get_probegroup()"
        return probes[0]

    def get_probes(self):
        probegroup = self.get_probegroup()
        return probegroup.probes

    def get_probegroup(self):
        arr = self.get_property("contact_vector")
        if arr is None:
            positions = self.get_property("location")
            if positions is None:
                raise ValueError("There is no Probe attached to this recording. Use set_probe(...) to attach one.")
            else:
                warn("There is no Probe attached to this recording. Creating a dummy one with contact positions")
                probe = self.create_dummy_probe_from_locations(positions)
                #  probe.create_auto_shape()
                probegroup = ProbeGroup()
                probegroup.add_probe(probe)
        else:
            probegroup = ProbeGroup.from_numpy(arr)
            for probe_index, probe in enumerate(probegroup.probes):
                contour = self.get_annotation(f"probe_{probe_index}_planar_contour")
                if contour is not None:
                    probe.set_planar_contour(contour)
        return probegroup

    def _extra_metadata_from_folder(self, folder):
        # load probe
        folder = Path(folder)
        if (folder / "probe.json").is_file():
            probegroup = read_probeinterface(folder / "probe.json")
            self.set_probegroup(probegroup, in_place=True)

    def _extra_metadata_to_folder(self, folder):
        # save probe
        if self.get_property("contact_vector") is not None:
            probegroup = self.get_probegroup()
            write_probeinterface(folder / "probe.json", probegroup)

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
        shape : str, default: default: "circle"
            Electrode shapes
        shape_params : dict, default: {"radius": 1}
            Shape parameters
        axes : "xy" | "yz" | "xz", default: "xy"
            If ndim is 3, indicates the axes that define the plane of the electrodes
        """
        probe = self.create_dummy_probe_from_locations(locations, shape=shape, shape_params=shape_params, axes=axes)
        self.set_probe(probe, in_place=True)

    def set_channel_locations(self, locations, channel_ids=None):
        if self.get_property("contact_vector") is not None:
            raise ValueError("set_channel_locations(..) destroys the probe description, prefer _set_probes(..)")
        self.set_property("location", locations, ids=channel_ids)

    def get_channel_locations(self, channel_ids=None, axes: str = "xy") -> np.ndarray:
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        channel_indices = self.ids_to_indices(channel_ids)
        contact_vector = self.get_property("contact_vector")
        if contact_vector is not None:
            # here we bypass the probe reconstruction so this works both for probe and probegroup
            ndim = len(axes)
            all_positions = np.zeros((contact_vector.size, ndim), dtype="float64")
            for i, dim in enumerate(axes):
                all_positions[:, i] = contact_vector[dim]
            positions = all_positions[channel_indices]
            return positions
        else:
            locations = self.get_property("location")
            if locations is None:
                raise Exception("There are no channel locations")
            locations = np.asarray(locations)[channel_indices]
            return select_axes(locations, axes)

    def has_3d_locations(self) -> bool:
        return self.get_property("location").shape[1] == 3

    def clear_channel_locations(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channel()
        else:
            n = len(channel_ids)
        locations = np.zeros((n, 2)) * np.nan
        self.set_property("location", locations, ids=channel_ids)

    def set_channel_groups(self, groups, channel_ids=None):
        if "probes" in self._annotations:
            warn("set_channel_groups() destroys the probe description. Using set_probe() is preferable")
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
        assert self.has_3d_locations, "The 'planarize' function needs a recording with 3d locations"
        assert len(axes) == 2, "You need to specify 2 dimensions (e.g. 'xy', 'zy')"

        probe2d = self.get_probe().to_2d(axes=axes)
        recording2d = self.clone()
        recording2d.set_probe(probe2d, in_place=True)

        return recording2d

    # utils
    def channel_slice(self, channel_ids, renamed_channel_ids=None):
        """
        Returns a new object with sliced channels.

        Parameters
        ----------
        channel_ids : np.array or list
            The list of channels to keep
        renamed_channel_ids : np.array or list, default: None
            A list of renamed channels

        Returns
        -------
        BaseRecordingSnippets
            The object with sliced channels
        """
        return self._channel_slice(channel_ids, renamed_channel_ids=renamed_channel_ids)

    def select_channels(self, channel_ids):
        """
        Returns a new object with sliced channels.

        Parameters
        ----------
        channel_ids : np.array or list
            The list of channels to keep

        Returns
        -------
        BaseRecordingSnippets
            The object with sliced channels
        """
        raise NotImplementedError

    def remove_channels(self, remove_channel_ids):
        """
        Returns a new object with removed channels.


        Parameters
        ----------
        remove_channel_ids : np.array or list
            The list of channels to remove

        Returns
        -------
        BaseRecordingSnippets
            The object with removed channels
        """
        return self._remove_channels(remove_channel_ids)

    def frame_slice(self, start_frame, end_frame):
        """
        Returns a new object with sliced frames.

        Parameters
        ----------
        start_frame : int
            The start frame
        end_frame : int
            The end frame

        Returns
        -------
        BaseRecordingSnippets
            The object with sliced frames
        """
        raise NotImplementedError

    def select_segments(self, segment_indices):
        """
        Return a new object with the segments specified by "segment_indices".

        Parameters
        ----------
        segment_indices : list of int
            List of segment indices to keep in the returned recording

        Returns
        -------
        BaseRecordingSnippets
            The onject with the selected segments
        """
        return self._select_segments(segment_indices)

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
            new_channel_ids = self.get_channel_ids()[inds]
            subrec = self.select_channels(new_channel_ids)
            subrec.set_annotation("split_by_property", value=property)
            if outputs == "list":
                recordings.append(subrec)
            elif outputs == "dict":
                recordings[value] = subrec
        return recordings
