from __future__ import annotations

import numpy as np
import json
from dataclasses import dataclass, field, astuple
from probeinterface import Probe
from pathlib import Path
from .sparsity import ChannelSparsity


@dataclass
class Templates:
    """
    A class to represent spike templates, which can be either dense or sparse.

    Parameters
    ----------
    templates_array : np.ndarray
        Array containing the templates data.
    sampling_frequency : float
        Sampling frequency of the templates.
    nbefore : int
        Number of samples before the spike peak.
    sparsity_mask : np.ndarray or None, default: None
        Boolean array indicating the sparsity pattern of the templates.
        If `None`, the templates are considered dense.
    channel_ids : np.ndarray, optional default: None
        Array of channel IDs. If `None`, defaults to an array of increasing integers.
    unit_ids : np.ndarray, optional default: None
        Array of unit IDs. If `None`, defaults to an array of increasing integers.
    probe: Probe, default: None
        A `probeinterface.Probe` object
    check_for_consistent_sparsity : bool, optional default: None
        When passing a sparsity_mask, this checks that the templates array is also sparse and that it matches the
        structure fo the sparsity_masl.

    The following attributes are available after construction:

    Attributes
    ----------
    num_units : int
        Number of units in the templates. Automatically determined from `templates_array`.
    num_samples : int
        Number of samples per template. Automatically determined from `templates_array`.
    num_channels : int
        Number of channels in the templates. Automatically determined from `templates_array` or `sparsity_mask`.
    nafter : int
        Number of samples after the spike peak. Calculated as `num_samples - nbefore - 1`.
    ms_before : float
        Milliseconds before the spike peak. Calculated from `nbefore` and `sampling_frequency`.
    ms_after : float
        Milliseconds after the spike peak. Calculated from `nafter` and `sampling_frequency`.
    sparsity : ChannelSparsity, optional
        Object representing the sparsity pattern of the templates. Calculated from `sparsity_mask`.
        If `None`, the templates are considered dense.
    """

    templates_array: np.ndarray
    sampling_frequency: float
    nbefore: int

    sparsity_mask: np.ndarray = None
    channel_ids: np.ndarray = None
    unit_ids: np.ndarray = None

    probe: Probe = None

    check_for_consistent_sparsity: bool = True

    num_units: int = field(init=False)
    num_samples: int = field(init=False)
    num_channels: int = field(init=False)

    nafter: int = field(init=False)
    ms_before: float = field(init=False)
    ms_after: float = field(init=False)
    sparsity: ChannelSparsity = field(init=False, default=None)

    def __post_init__(self):
        self.num_units, self.num_samples = self.templates_array.shape[:2]
        if self.sparsity_mask is None:
            self.num_channels = self.templates_array.shape[2]
        else:
            self.num_channels = self.sparsity_mask.shape[1]

        if self.probe is not None:
            assert isinstance(self.probe, Probe), "'probe' must be a probeinterface.Probe object"

        # Time and frames domain information
        self.nafter = self.num_samples - self.nbefore
        self.ms_before = self.nbefore / self.sampling_frequency * 1000
        self.ms_after = self.nafter / self.sampling_frequency * 1000

        # Initialize sparsity object
        if self.channel_ids is None:
            self.channel_ids = np.arange(self.num_channels)
        if self.unit_ids is None:
            self.unit_ids = np.arange(self.num_units)
        if self.sparsity_mask is not None:
            self.sparsity = ChannelSparsity(
                mask=self.sparsity_mask,
                unit_ids=self.unit_ids,
                channel_ids=self.channel_ids,
            )

            # Test that the templates are sparse if a sparsity mask is passed
            if self.check_for_consistent_sparsity:
                if not self._are_passed_templates_sparse():
                    raise ValueError("Sparsity mask passed but the templates are not sparse")

    def get_one_template_dense(self, unit_index):
        if self.sparsity is None:
            template = self.templates_array[unit_index, :, :]
        else:
            sparse_template = self.templates_array[unit_index, :, :]
            template = self.sparsity.densify_waveforms(waveforms=sparse_template, unit_id=self.unit_ids[unit_index])
            # dense_waveforms[unit_index, ...] = self.sparsity.densify_waveforms(waveforms=waveforms, unit_id=unit_id)
        return template

    def get_dense_templates(self) -> np.ndarray:
        # Assumes and object without a sparsity mask already has dense templates
        if self.sparsity is None:
            return self.templates_array

        densified_shape = (self.num_units, self.num_samples, self.num_channels)
        dense_waveforms = np.zeros(shape=densified_shape, dtype=self.templates_array.dtype)

        for unit_index, unit_id in enumerate(self.unit_ids):
            # waveforms = self.templates_array[unit_index, ...]
            # dense_waveforms[unit_index, ...] = self.sparsity.densify_waveforms(waveforms=waveforms, unit_id=unit_id)
            dense_waveforms[unit_index, ...] = self.get_one_template_dense(unit_index)

        return dense_waveforms

    def are_templates_sparse(self) -> bool:
        return self.sparsity is not None

    def _are_passed_templates_sparse(self) -> bool:
        """
        Tests if the templates passed to the init constructor are sparse
        """
        are_templates_sparse = True
        for unit_index, unit_id in enumerate(self.unit_ids):
            waveforms = self.templates_array[unit_index, ...]
            are_templates_sparse = self.sparsity.are_waveforms_sparse(waveforms, unit_id=unit_id)
            if not are_templates_sparse:
                return False

        return are_templates_sparse

    def to_dict(self):
        return {
            "templates_array": self.templates_array,
            "sparsity_mask": None if self.sparsity_mask is None else self.sparsity_mask,
            "channel_ids": self.channel_ids,
            "unit_ids": self.unit_ids,
            "sampling_frequency": self.sampling_frequency,
            "nbefore": self.nbefore,
            "probe": self.probe.to_dict() if self.probe is not None else None,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            templates_array=np.asarray(data["templates_array"]),
            sparsity_mask=None if data["sparsity_mask"] is None else np.asarray(data["sparsity_mask"]),
            channel_ids=np.asarray(data["channel_ids"]),
            unit_ids=np.asarray(data["unit_ids"]),
            sampling_frequency=data["sampling_frequency"],
            nbefore=data["nbefore"],
            probe=data["probe"] if data["probe"] is None else Probe.from_dict(data["probe"]),
        )

    def add_templates_to_zarr_group(self, zarr_group: "zarr.Group") -> None:
        """
        Adds a serialized version of the object to a given Zarr group.

        It is the inverse of the `from_zarr_group` method.

        Parameters
        ----------
        zarr_group : zarr.Group
            The Zarr group to which the template object will be serialized.

        Notes
        -----
        This method will create datasets within the Zarr group for `templates_array`,
        `channel_ids`, and `unit_ids`. It will also add `sampling_frequency` and `nbefore`
        as attributes to the group. If `sparsity_mask` and `probe` are not None, they will
        be included as a dataset and a subgroup, respectively.

        The `templates_array` dataset is saved with a chunk size that has a single unit per chunk
        to optimize read/write operations for individual units.
        """

        # Saves one chunk per unit
        arrays_chunk = (1, None, None)
        zarr_group.create_dataset("templates_array", data=self.templates_array, chunks=arrays_chunk)
        zarr_group.create_dataset("channel_ids", data=self.channel_ids)
        zarr_group.create_dataset("unit_ids", data=self.unit_ids)

        zarr_group.attrs["sampling_frequency"] = self.sampling_frequency
        zarr_group.attrs["nbefore"] = self.nbefore

        if self.sparsity_mask is not None:
            zarr_group.create_dataset("sparsity_mask", data=self.sparsity_mask)

        if self.probe is not None:
            probe_group = zarr_group.create_group("probe")
            self.probe.add_probe_to_zarr_group(probe_group)

    def to_zarr(self, folder_path: str | Path) -> None:
        """
        Saves the object's data to a Zarr file in the specified folder.

        Use the `add_templates_to_zarr_group` method to serialize the object to a Zarr group and then
        save the group to a Zarr file.

        Parameters
        ----------
        folder_path : str | Path
            The path to the folder where the Zarr data will be saved.

        """
        import zarr

        zarr_group = zarr.open_group(folder_path, mode="w")

        self.add_templates_to_zarr_group(zarr_group)

    @classmethod
    def from_zarr_group(cls, zarr_group: "zarr.Group") -> "Templates":
        """
        Loads an instance of the class from an open Zarr group.

        This is the inverse of the `add_templates_to_zarr_group` method.

        Parameters
        ----------
        zarr_group : zarr.Group
            The Zarr group from which to load the instance.

        Returns
        -------
        Templates
            An instance of Templates populated with the data from the Zarr group.

        Notes
        -----
        This method assumes the Zarr group has the same structure as the one created by
        the `add_templates_to_zarr_group` method.

        """
        templates_array = zarr_group["templates_array"]
        channel_ids = zarr_group["channel_ids"]
        unit_ids = zarr_group["unit_ids"]
        sampling_frequency = zarr_group.attrs["sampling_frequency"]
        nbefore = zarr_group.attrs["nbefore"]

        sparsity_mask = None
        if "sparsity_mask" in zarr_group:
            sparsity_mask = zarr_group["sparsity_mask"]

        probe = None
        if "probe" in zarr_group:
            probe = Probe.from_zarr_group(zarr_group["probe"])

        return cls(
            templates_array=templates_array,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            sparsity_mask=sparsity_mask,
            channel_ids=channel_ids,
            unit_ids=unit_ids,
            probe=probe,
        )

    @staticmethod
    def from_zarr(folder_path: str | Path) -> "Templates":
        """
        Deserialize the Templates object from a Zarr file located at the given folder path.

        Parameters
        ----------
        folder_path : str | Path
            The path to the folder where the Zarr file is located.

        Returns
        -------
        Templates
            An instance of Templates initialized with data from the Zarr file.
        """
        import zarr

        zarr_group = zarr.open_group(folder_path, mode="r")

        return Templates.from_zarr_group(zarr_group)

    def to_json(self):
        from spikeinterface.core.core_tools import SIJsonEncoder

        return json.dumps(self.to_dict(), cls=SIJsonEncoder)

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))

    def __eq__(self, other):
        """
        Necessary to compare templates because they naturally compare objects by equality of their fields
        which is not possible for numpy arrays. Therefore, we override the __eq__ method to compare each numpy arrays
        using np.array_equal instead
        """
        if not isinstance(other, Templates):
            return False

        # Convert the instances to tuples
        self_tuple = astuple(self)
        other_tuple = astuple(other)

        # Compare each field
        for s_field, o_field in zip(self_tuple, other_tuple):
            if isinstance(s_field, np.ndarray):
                if not np.array_equal(s_field, o_field):
                    return False

            # Compare ChannelSparsity by its mask, unit_ids and channel_ids.
            # Maybe ChannelSparsity should have its own __eq__ method
            elif isinstance(s_field, ChannelSparsity):
                if not isinstance(o_field, ChannelSparsity):
                    return False

                # Compare ChannelSparsity by its mask, unit_ids and channel_ids
                if not np.array_equal(s_field.mask, o_field.mask):
                    return False
                if not np.array_equal(s_field.unit_ids, o_field.unit_ids):
                    return False
                if not np.array_equal(s_field.channel_ids, o_field.channel_ids):
                    return False
            else:
                if s_field != o_field:
                    return False

        return True

    def get_channel_locations(self):
        assert self.probe is not None, "Templates.get_channel_locations() needs a probe to be set"
        channel_locations = self.probe.contact_positions
        return channel_locations
