from pathlib import Path
from typing import Literal
import warnings
import numpy as np

from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface, select_axes

from .base import BaseExtractor
from .recording_tools import _set_group_property_based_on_probegroup, check_probe_do_not_overlap

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

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get("is_filtered", False)

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
    ) -> "BaseRecordingSnippets":
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
    ) -> "BaseRecordingSnippets":
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
            warn(
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

    def _extra_metadata_copy(self, other):
        if self._probegroup is not None:
            other._probegroup = self._probegroup.copy()

    def _extra_metadata_from_folder(self, folder):
        # load probe from folder
        # Note: we don't need any fix for legacy probegroups, since the
        # set_probegroup() method will handle the device_channel_indices
        # sorting and global contact order
        folder = Path(folder)
        probe_file = folder / "probegroup.json"
        legacy_probe_file = folder / "probe.json"
        if probe_file.is_file():
            probegroup = read_probeinterface(probe_file)
            self.set_probegroup(probegroup)
        elif legacy_probe_file.is_file():
            probegroup = read_probeinterface(legacy_probe_file)
            self.set_probegroup(probegroup)

        # remove "contact_vector" property if present as it is not needed anymore
        if "contact_vector" in self.get_property_keys():
            self.delete_property("contact_vector")

    def _extra_metadata_to_folder(self, folder):
        # save probe
        if self.has_probe():
            probegroup = self.get_probegroup()
            write_probeinterface(folder / "probegroup.json", probegroup)

    def _extra_metadata_from_dict(self, dump_dict):
        # load probe and hanlde backward-compatibility with legacy "contact_vector"/"location" property
        if "probegroup" in dump_dict:
            # this is for SI>=0.105.0
            probegroup = dump_dict["probegroup"]
            self._probegroup = ProbeGroup.from_dict(probegroup)

    def _extra_metadata_to_dict(self, dump_dict):
        # save probe
        if self.has_probe():
            probegroup = self.get_probegroup()
            dump_dict["probegroup"] = probegroup.to_dict()

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
        recording2d.set_probe(probe2d)

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
