import numpy as np
import json
from dataclasses import dataclass, field, astuple
from .sparsity import ChannelSparsity


@dataclass(kw_only=True)
class Templates:
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
        self.nafter = self.num_samples - self.nbefore - 1
        self.ms_before = self.nbefore / self.sampling_frequency * 1000
        self.ms_after = self.nafter / self.sampling_frequency * 1000
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

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))

    def __eq__(self, other):
        """Necessary to compare arrays"""
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

            elif isinstance(s_field, ChannelSparsity):
                if not isinstance(o_field, ChannelSparsity):
                    return False

                # (maybe ChannelSparsity should have its own __eq__ method)
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
