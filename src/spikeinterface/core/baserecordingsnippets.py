import copy
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
        return self._probegroup is not None

    def has_channel_location(self) -> bool:
        return self.has_probe() or "location" in self.get_property_keys()

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get("is_filtered", False)

    def set_probe(self, probe, group_mode="auto", in_place=False):
        assert isinstance(probe, Probe), "must give Probe"
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self._set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probegroup(self, probegroup, group_mode="auto", in_place=False):
        return self._set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def _set_probes(self, probe_or_probegroup, group_mode="auto", in_place=False):
        """
        Attach a Probe, ProbeGroup, or list of Probe to the recording.

        The probegroup is stored by reference without mutation. The contact-to-channel
        mapping is built from each probe's current `device_channel_indices` and stored
        on the recording as two per-channel properties, `probe_id` and `contact_id`.
        `location` and `group` are also written as properties for backward compatibility.

        If the probegroup wires only a subset of the recording's channels, the recording
        is sliced via `select_channels` to keep only the wired channels (preserving the
        historical attach semantics).

        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording.
        in_place: bool, default: False
            If True, attach to self in place.

        Returns
        -------
        sub_recording: BaseRecording
            The recording with the probegroup attached.
        """
        # normalize input to a ProbeGroup
        if isinstance(probe_or_probegroup, Probe):
            probegroup = ProbeGroup()
            probegroup.add_probe(probe_or_probegroup)
        elif isinstance(probe_or_probegroup, ProbeGroup):
            probegroup = probe_or_probegroup
        elif isinstance(probe_or_probegroup, list):
            assert all(isinstance(e, Probe) for e in probe_or_probegroup)
            probegroup = ProbeGroup()
            for probe in probe_or_probegroup:
                probegroup.add_probe(probe)
        else:
            raise ValueError("must give Probe or ProbeGroup or list of Probe")

        if len(probegroup.probes) > 1:
            check_probe_do_not_overlap(probegroup.probes)

        assert all(
            probe.device_channel_indices is not None for probe in probegroup.probes
        ), "Probe must have device_channel_indices"

        # ensure every probe has a stable probe_id and every contact has a contact_id;
        # auto-generate only when missing so user-assigned ids survive
        if not any("probe_id" in p.annotations for p in probegroup.probes):
            probegroup.auto_generate_probe_ids()
        if any(p.contact_ids is None for p in probegroup.probes):
            probegroup.auto_generate_contact_ids()

        # collect, per recording channel (by position in self.channel_ids), which
        # (probe_id, contact_id) pair wires into it. Unwired channels stay as None.
        num_channels = self.get_num_channels()
        probe_id_col = [None] * num_channels
        contact_id_col = [None] * num_channels
        for probe in probegroup.probes:
            probe_id = probe.annotations["probe_id"]
            dci = np.asarray(probe.device_channel_indices)
            for contact_idx, device_idx in enumerate(dci):
                if device_idx < 0:
                    continue  # unconnected contact; skip
                if device_idx >= num_channels:
                    raise ValueError(
                        f"device_channel_indices value {device_idx} is out of range; "
                        f"recording has {num_channels} channels."
                    )
                if probe_id_col[device_idx] is not None:
                    raise ValueError(f"channel at index {device_idx} is wired to more than one contact.")
                probe_id_col[device_idx] = probe_id
                contact_id_col[device_idx] = probe.contact_ids[contact_idx]

        # Reorder the recording's channels to match the probe's device_channel_indices
        # order (smallest dci first). Also drops any channels that are unwired. This
        # matches the historical `set_probe` behaviour: after attach, recording channel
        # i corresponds to the probe contact whose dci was the i-th smallest.
        wired_dci_pairs = []  # (device_idx, position_in_probe_id_col)
        for i, pid in enumerate(probe_id_col):
            if pid is not None:
                wired_dci_pairs.append(i)  # position == original device_idx since we indexed by it
        # sort by device_idx (which equals the position already)
        ordered_positions = sorted(wired_dci_pairs)
        original_channel_ids = self.get_channel_ids()
        new_channel_ids = original_channel_ids[ordered_positions]

        if in_place:
            if not np.array_equal(new_channel_ids, original_channel_ids):
                raise Exception("set_probe(in_place=True) must have all channel indices")
            target = self
        else:
            if np.array_equal(new_channel_ids, original_channel_ids):
                target = self.clone()
            else:
                target = self.select_channels(new_channel_ids)
        # re-key probe_id_col / contact_id_col into the (possibly reordered) target
        probe_id_col = [probe_id_col[i] for i in ordered_positions]
        contact_id_col = [contact_id_col[i] for i in ordered_positions]

        # attach probegroup; the wiring lives as a (num_channels, 2) per-channel
        # string property `wiring` with column 0 = probe_id, column 1 = contact_id.
        # This is the same pattern as `location` (2D property per channel) and rides
        # on SI's existing property plumbing (copy_metadata, concat, serialization).
        target._probegroup = probegroup

        # handle the degenerate empty-wiring case (probe with all dci=-1)
        if len(probe_id_col) == 0:
            ndim = probegroup.ndim
            target.set_property("wiring", np.zeros((0, 2), dtype="U64"))
            target.set_property("location", np.zeros((0, ndim), dtype="float64"))
            target.set_property("group", np.zeros(0, dtype="int64"))
            return target

        wiring = np.column_stack(
            [
                np.asarray(probe_id_col, dtype="U64"),
                np.asarray(contact_id_col, dtype="U64"),
            ]
        )
        target.set_property("wiring", wiring)

        # write `location` and `group` as compatibility mirrors of the canonical
        # probegroup + _channel_to_contact mapping. group_mode is consulted here to
        # match the pre-strong-preserve API; callers that pass "by_probe", "by_shank"
        # etc. get the same partitioning as before.
        ndim = probegroup.ndim
        probes_by_id = {p.annotations["probe_id"]: p for p in probegroup.probes}
        has_shank = any(p.shank_ids is not None for p in probegroup.probes)
        has_side = any(p.contact_sides is not None for p in probegroup.probes)
        if group_mode == "auto":
            keys_template = ["probe"] + (["shank"] if has_shank else []) + (["side"] if has_side else [])
        elif group_mode == "by_probe":
            keys_template = ["probe"]
        elif group_mode == "by_shank":
            assert has_shank, "shank_ids is None in probe, you cannot group by shank"
            keys_template = ["probe", "shank"]
        elif group_mode == "by_side":
            assert has_side, "contact_sides is None in probe, you cannot group by side"
            keys_template = ["probe"] + (["shank"] if has_shank else []) + ["side"]
        else:
            raise ValueError(f"unknown group_mode {group_mode!r}")

        wired_positions = list(range(len(probe_id_col)))
        locations = np.zeros((len(wired_positions), ndim), dtype="float64")
        group_keys_per_channel = []
        for i, (pid, cid) in enumerate(zip(probe_id_col, contact_id_col)):
            probe = probes_by_id[pid]
            contact_idx = int(np.where(np.asarray(probe.contact_ids) == cid)[0][0])
            locations[i] = probe.contact_positions[contact_idx, :ndim]
            key = []
            for k in keys_template:
                if k == "probe":
                    key.append(pid)
                elif k == "shank" and probe.shank_ids is not None:
                    key.append(probe.shank_ids[contact_idx])
                elif k == "side" and probe.contact_sides is not None:
                    key.append(probe.contact_sides[contact_idx])
            group_keys_per_channel.append(tuple(key))

        unique_keys = list(dict.fromkeys(group_keys_per_channel))
        key_to_int = {k: i for i, k in enumerate(unique_keys)}
        groups = np.array([key_to_int[k] for k in group_keys_per_channel], dtype="int64")

        target.set_property("location", locations)
        target.set_property("group", groups)
        return target

    def get_probe(self):
        probes = self.get_probes()
        assert len(probes) == 1, "there are several probe use .get_probes() or get_probegroup()"
        return probes[0]

    def get_probes(self):
        probegroup = self.get_probegroup()
        return probegroup.probes

    def get_probegroup(self):
        if self._probegroup is None:
            # Backwards-compat fallback: pre-migration get_probegroup synthesised a dummy
            # probe from the "location" property when no probe had been attached. Callers
            # (e.g. sparsity.py) rely on this for recordings that have locations but no
            # probe.
            positions = self.get_property("location")
            if positions is None:
                raise ValueError("There is no Probe attached to this recording. Use set_probe(...) to attach one.")
            warn("There is no Probe attached to this recording. Creating a dummy one with contact positions")
            probe = self.create_dummy_probe_from_locations(positions)
            pg = ProbeGroup()
            pg.add_probe(probe)
            return copy.deepcopy(pg)

        # Build a channel-ordered view of the stored probegroup for the public getter.
        # Strong-preserve keeps each probe intact on `_probegroup`; here we slice it down
        # to the contacts that actually appear in this recording, in channel order, with
        # device_channel_indices = arange(N). The returned object matches the
        # pre-strong-preserve `get_probe()` semantic.
        wiring = self.get_property("wiring")
        if wiring is None:
            return copy.deepcopy(self._probegroup)

        # map (probe_id, contact_id) to the global contact index in the stored probegroup
        contact_id_to_global = {}
        offset = 0
        for probe in self._probegroup.probes:
            pid = probe.annotations["probe_id"]
            for cid in probe.contact_ids:
                contact_id_to_global[(pid, cid)] = offset
                offset += 1

        global_indices = [contact_id_to_global[(pid, cid)] for pid, cid in wiring]
        view = self._probegroup.get_slice(np.asarray(global_indices, dtype="int64"))
        view.set_global_device_channel_indices(np.arange(len(global_indices), dtype="int64"))
        return view

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
        if self.has_probe():
            raise ValueError("set_channel_locations(..) destroys the probe description, prefer _set_probes(..)")
        self.set_property("location", locations, ids=channel_ids)

    def get_channel_locations(self, channel_ids=None, axes: str = "xy") -> np.ndarray:
        if channel_ids is None:
            channel_ids = self.get_channel_ids()

        if self.has_probe():
            # resolve each channel via the `wiring` property (column 0 = probe_id,
            # column 1 = contact_id) and look up the contact's position on the
            # corresponding probe
            wiring = self.get_property("wiring", ids=channel_ids)
            probes_by_id = {p.annotations["probe_id"]: p for p in self._probegroup.probes}
            axis_index = {"x": 0, "y": 1, "z": 2}
            ndim = len(axes)
            locations = np.zeros((len(channel_ids), ndim), dtype="float64")
            for i, (probe_id, contact_id) in enumerate(wiring):
                probe = probes_by_id[probe_id]
                contact_idx = int(np.where(np.asarray(probe.contact_ids) == contact_id)[0][0])
                for j, axis in enumerate(axes):
                    locations[i, j] = probe.contact_positions[contact_idx, axis_index[axis]]
            return locations

        # fallback for recordings that have a "location" property but no attached probegroup
        channel_indices = self.ids_to_indices(channel_ids)
        locations = self.get_property("location")
        if locations is None:
            raise Exception("There are no channel locations")
        locations = np.asarray(locations)[channel_indices]
        return select_axes(locations, axes)

    def has_3d_locations(self) -> bool:
        if self.has_probe():
            return self._probegroup.ndim == 3
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
        # when a probe is attached, derive groups on the fly from the `wiring`
        # property + probegroup state (probe_id + shank_ids + contact_sides)
        if self.has_probe():
            if channel_ids is None:
                channel_ids = self.get_channel_ids()
            wiring = self.get_property("wiring", ids=channel_ids)
            probes_by_id = {p.annotations["probe_id"]: p for p in self._probegroup.probes}
            has_shank = any(p.shank_ids is not None for p in self._probegroup.probes)
            has_side = any(p.contact_sides is not None for p in self._probegroup.probes)

            group_keys = []
            for probe_id, contact_id in wiring:
                probe = probes_by_id[probe_id]
                key = [probe_id]
                if has_shank or has_side:
                    contact_idx = int(np.where(np.asarray(probe.contact_ids) == contact_id)[0][0])
                    if has_shank and probe.shank_ids is not None:
                        key.append(probe.shank_ids[contact_idx])
                    if has_side and probe.contact_sides is not None:
                        key.append(probe.contact_sides[contact_idx])
                group_keys.append(tuple(key))

            unique_keys = list(dict.fromkeys(group_keys))
            key_to_int = {k: i for i, k in enumerate(unique_keys)}
            return np.array([key_to_int[k] for k in group_keys], dtype="int64")

        # fallback: read a stored "group" property (recordings without a probe)
        return self.get_property("group", ids=channel_ids)

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
