import numpy as np
import json
from dataclasses import dataclass, field, astuple
from .sparsity import ChannelSparsity


@dataclass(kw_only=True)
class Templates:
    """
    A class to represent spike templates, which can be either dense or sparse.

    Attributes
    ----------
    templates_array : np.ndarray
        Array containing the templates data.
    sampling_frequency : float
        Sampling frequency of the templates.
    nbefore : int
        Number of samples before the spike peak.
    sparsity_mask : np.ndarray, optional
        Binary array indicating the sparsity pattern of the templates.
        If `None`, the templates are considered dense.
    channel_ids : np.ndarray, optional
        Array of channel IDs. If `None`, defaults to an array of increasing integers.
    unit_ids : np.ndarray, optional
        Array of unit IDs. If `None`, defaults to an array of increasing integers.
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
            if not self._are_passed_templates_sparse():
                raise ValueError("Sparsity mask passed but the templates are not sparse")

    def to_dict(self):
        return {
            "templates_array": self.templates_array.tolist(),
            "sparsity_mask": None if self.sparsity_mask is None else self.sparsity_mask.tolist(),
            "channel_ids": self.channel_ids.tolist(),
            "unit_ids": self.unit_ids.tolist(),
            "sampling_frequency": self.sampling_frequency,
            "nbefore": self.nbefore,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            templates_array=np.array(data["templates_array"]),
            sparsity_mask=None if data["sparsity_mask"] is None else np.array(data["sparsity_mask"]),
            channel_ids=np.array(data["channel_ids"]),
            unit_ids=np.array(data["unit_ids"]),
            sampling_frequency=data["sampling_frequency"],
            nbefore=data["nbefore"],
        )

    def get_dense_templates(self) -> np.ndarray:
        # Assumes and object without a sparsity mask already has dense templates
        if self.sparsity is None:
            return self.templates_array

        densified_shape = (self.num_units, self.num_samples, self.num_channels)
        dense_waveforms = np.zeros(shape=densified_shape, dtype=self.templates_array.dtype)

        for unit_index, unit_id in enumerate(self.unit_ids):
            waveforms = self.templates_array[unit_index, ...]
            dense_waveforms[unit_index, ...] = self.sparsity.densify_waveforms(waveforms=waveforms, unit_id=unit_id)

        return dense_waveforms

    def get_sparse_templates(self) -> np.ndarray:
        # Objects without sparsity mask don't have sparsity and therefore can't return sparse templates
        if self.sparsity is None:
            raise ValueError("Can't return sparse templates without passing a sparsity mask")

        max_num_active_channels = self.sparsity.max_num_active_channels
        sparisfied_shape = (self.num_units, self.num_samples, max_num_active_channels)
        sparse_waveforms = np.zeros(shape=sparisfied_shape, dtype=self.templates_array.dtype)
        for unit_index, unit_id in enumerate(self.unit_ids):
            waveforms = self.templates_array[unit_index, ...]
            sparse_waveforms[unit_index, ...] = self.sparsity.sparsify_waveforms(waveforms=waveforms, unit_id=unit_id)

        return sparse_waveforms

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

    def to_json(self):
        return json.dumps(self.to_dict())

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
