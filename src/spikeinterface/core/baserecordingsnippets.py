from pathlib import Path
import warnings
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
        self._probegroup = None

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

    def has_probe(self) -> bool:
        if self._probegroup is None and self.get_property("contact_vector") is not None:
            # if contact_vector is present we can reconstruct the probe
            self._probegroup = self._build_probegroup_from_properties()
        return self._probegroup is not None

    def has_3d_probe(self) -> bool:
        if self.has_probe():
            probe = self.get_probegroup().probes[0]
            return probe.ndim == 3
        else:
            return False

    def has_channel_location(self) -> bool:
        return self.has_probe()

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get("is_filtered", False)

    def reset_probe(self):
        """
        Removes probe information
        """
        self._probegroup = None

    def set_probe(self, probe, group_mode="auto", in_place=False):
        """
        Attach a list of Probe object to a recording.

        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording
        group_mode: "auto" | "by_probe" | "by_shank" | "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks and two sides are present.
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

    def set_probegroup(self, probegroup, group_mode="auto", in_place=False, raise_if_overlapping_probes=True):
        """
        Attach a ProbeGroup to a recording.
        For this ProbeGroup.get_global_device_channel_indices() is used to link contacts to recording channels.
        If some contacts of the probe group are not connected (device_channel_indices=-1)
        then the recording is "sliced" and only connected channel are kept.

        The probe group order is not kept. Channel ids are re-ordered to match the channel_ids of the recording.

        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording
        group_mode: "auto" | "by_probe" | "by_shank" | "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks and two sides are present.
        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)
        raise_if_overlapping_probes: bool
            If True, raises an error if the probes overlap. If False, it will just warn

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        return self._set_probes(
            probegroup,
            group_mode=group_mode,
            in_place=in_place,
            raise_if_overlapping_probes=raise_if_overlapping_probes,
        )

    def _set_probes(self, probe_or_probegroup, group_mode="auto", in_place=False, raise_if_overlapping_probes=True):
        """
        Attach a list of Probe objects or a ProbeGroup to a recording.
        For this Probe.device_channel_indices is used to link contacts to recording channels.
        If some contacts of the Probe are not connected (device_channel_indices=-1)
        then the recording is "sliced" and only connected channel are kept.

        The probe order is not kept. Channel ids are re-ordered to match the channel_ids of the recording.


        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probes, ProbeGroup, or dict
            The probe(s) to be attached to the recording
        group_mode: "auto" | "by_probe" | "by_shank" | "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks and two sides are present.
        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)
        raise_if_overlapping_probes: bool
            If True, raises an error if the probes overlap. If False, it will just warn

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        assert group_mode in (
            "auto",
            "by_probe",
            "by_shank",
            "by_side",
        ), "'group_mode' can be 'auto' 'by_probe' 'by_shank' or 'by_side'"

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
        elif isinstance(probe_or_probegroup, dict):
            probegroup = ProbeGroup.from_dict(probe_or_probegroup)
        else:
            raise ValueError("must give Probe or ProbeGroup or list of Probe")

        # check that the probe do not overlap
        num_probes = len(probegroup.probes)
        if num_probes > 1 and raise_if_overlapping_probes:
            check_probe_do_not_overlap(probegroup.probes)

        # handle not connected channels
        assert all(
            probe.device_channel_indices is not None for probe in probegroup.probes
        ), "Probe must have device_channel_indices"

        # TODO: add get_slice for probegroup to handle not connected channels
        probe_as_numpy_array = probegroup.to_numpy(complete=True)
        device_channel_indices = probegroup.get_global_device_channel_indices()["device_channel_indices"]
        keep = device_channel_indices >= 0
        if np.any(~keep):
            warn("The given probes have unconnected contacts: they are removed")
        device_channel_indices = device_channel_indices[keep]
        probe_as_numpy_array = probe_as_numpy_array[keep]
        if len(device_channel_indices) > 0:
            probegroup = probegroup.get_slice(device_channel_indices)
            order = np.argsort(device_channel_indices)
            device_channel_indices = device_channel_indices[order]
            probegroup.set_global_device_channel_indices(np.arange(len(device_channel_indices)))

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
        else:
            warn("No connected channel in the probe! The probe will be attached but no channel will be selected.")
            probegroup = ProbeGroup()  # empty probegroup

        new_channel_ids = self.channel_ids[device_channel_indices]

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

        # # create a vector that handle all contacts in property
        # sub_recording.set_property("contact_vector", probe_as_numpy_array, ids=None)
        sub_recording._probegroup = probegroup

        # handle groups
        has_shank_id = "shank_ids" in probe_as_numpy_array.dtype.fields
        has_contact_side = "contact_sides" in probe_as_numpy_array.dtype.fields
        if group_mode == "auto":
            group_keys = ["probe_index"]
            if has_shank_id:
                group_keys += ["shank_ids"]
            if has_contact_side:
                group_keys += ["contact_sides"]
        elif group_mode == "by_probe":
            group_keys = ["probe_index"]
        elif group_mode == "by_shank":
            assert has_shank_id, "shank_ids is None in probe, you cannot group by shank"
            group_keys = ["probe_index", "shank_ids"]
        elif group_mode == "by_side":
            assert has_contact_side, "contact_sides is None in probe, you cannot group by side"
            if has_shank_id:
                group_keys = ["probe_index", "shank_ids", "contact_sides"]
            else:
                group_keys = ["probe_index", "contact_sides"]
        groups = np.zeros(probe_as_numpy_array.size, dtype="int64")
        unique_keys = np.unique(probe_as_numpy_array[group_keys])
        for group, a in enumerate(unique_keys):
            mask = np.ones(probe_as_numpy_array.size, dtype=bool)
            for k in group_keys:
                mask &= probe_as_numpy_array[k] == a[k]
            groups[mask] = group
        sub_recording.set_property("group", groups, ids=None)

        return sub_recording

    def get_probe(self):
        probes = self.get_probes()
        assert len(probes) == 1, "There are several probe use .get_probes() or get_probegroup()"
        return probes[0]

    def get_probes(self):
        probegroup = self.get_probegroup()
        return probegroup.probes

    def get_probegroup(self):
        if self._probegroup is not None:
            return self._probegroup
        else:  # Backward compatibility: if contact_vector is present we reconstruct the probe, otherwise we look for
            probegroup = self._build_probegroup_from_properties()
            if probegroup is None:
                raise ValueError("There is no Probe attached to this recording. Use set_probe(...) to attach one.")
            self._probegroup = probegroup
            return probegroup

    def _build_probegroup_from_properties(self):
        # location and create a dummy probe
        arr = self.get_property("contact_vector")
        if arr is None:
            positions = self.get_property("location")
            if positions is None:
                return None
            else:
                warn("There is no Probe attached to this recording. Creating a dummy one with contact positions")
                probe = self.create_dummy_probe_from_locations(positions)
                #  probe.create_auto_shape()
                probegroup = ProbeGroup()
                probegroup.add_probe(probe)
        else:
            probegroup = ProbeGroup.from_numpy(arr)

            if "probes_info" in self.get_annotation_keys():
                probes_info = self.get_annotation("probes_info")
                for probe, probe_info in zip(probegroup.probes, probes_info):
                    probe.annotations = probe_info

            for probe_index, probe in enumerate(probegroup.probes):
                contour = self.get_annotation(f"probe_{probe_index}_planar_contour")
                if contour is not None:
                    probe.set_planar_contour(contour)
                self.delete_annotation(f"probe_{probe_index}_planar_contour")
            # delete contact_vector as it is not needed anymore
            self.delete_property("contact_vector")
        return probegroup

    def _extra_metadata_copy(self, other):
        if self._probegroup is not None:
            other._probegroup = self._probegroup.copy()

    def _extra_metadata_from_folder(self, folder):
        # load probe
        folder = Path(folder)
        if (folder / "probe.json").is_file():
            probegroup = read_probeinterface(folder / "probe.json")
            self.set_probegroup(probegroup, in_place=True)

    def _extra_metadata_to_folder(self, folder):
        # save probe
        if self.has_probe():
            probegroup = self.get_probegroup()
            write_probeinterface(folder / "probe.json", probegroup)

    def _extra_metadata_from_dict(self, dump_dict):
        # load probe
        if "probegroup" in dump_dict:
            probegroup = dump_dict["probegroup"]
            self.set_probegroup(probegroup, in_place=True)

    def _extra_metadata_to_dict(self, dump_dict):
        # save probe
        if self.has_probe():
            probegroup = self.get_probegroup()
            dump_dict["probegroup"] = probegroup

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
        self.set_probe(probe, in_place=True)

    def set_channel_locations(self, locations, channel_ids=None):
        warnings.warn(
            (
                "set_channel_locations() is deprecated and will be removed in version 0.106.0. "
                "If you want to set probe information, use `set_dummy_probe_from_locations()`."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_dummy_probe_from_locations(locations, axes="xy")

    def get_channel_locations(self, channel_ids=None, axes: str = "xy") -> np.ndarray:
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
        probe = self.get_probegroup().probes[0]
        return probe.ndim == 3

    def clear_channel_locations(self, channel_ids=None):
        warnings.warn(
            (
                "clear_channel_locations() is deprecated and will be removed in version 0.106.0. "
                "If you want to remove probe information, use `reset_probe()`."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        self.reset_probe()

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
        assert self.has_3d_probe(), "The 'planarize' function needs a recording with 3d locations"
        assert len(axes) == 2, "You need to specify 2 dimensions (e.g. 'xy', 'zy')"

        probe2d = self.get_probe().to_2d(axes=axes)
        recording2d = self.clone()
        recording2d.set_probe(probe2d, in_place=True)

        return recording2d

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
            new_channel_ids = self.channel_ids[inds]
            subrec = self.select_channels(new_channel_ids)
            subrec.set_annotation("split_by_property", value=property)
            if outputs == "list":
                recordings.append(subrec)
            elif outputs == "dict":
                recordings[value] = subrec
        return recordings
