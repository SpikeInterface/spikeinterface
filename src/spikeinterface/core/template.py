import json
from dataclasses import dataclass, field

import numpy as np

from spikeinterface.core.sparsity import ChannelSparsity


@dataclass
class Templates:
    templates_array: np.ndarray
    sparsity: ChannelSparsity = None
    num_units: int = field(init=False)
    num_samples: int = field(init=False)
    num_channels: int = field(init=False)

    def __post_init__(self):
        self.num_units, self.num_samples, self.num_channels = self.templates_array.shape

    # Implementing the slicing/indexing behavior as numpy
    def __getitem__(self, index):
        return self.templates_array[index]

    def __array__(self):
        return self.templates_array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> np.ndarray:
        # Replace any Templates instances with their ndarray representation
        inputs = tuple(inp.templates_array if isinstance(inp, Templates) else inp for inp in inputs)

        # Apply the ufunc on the transformed inputs
        result = getattr(ufunc, method)(*inputs, **kwargs)

        return result

    def to_dict(self):
        sparsity = self.sparsity.to_dict() if self.sparsity is not None else None
        return {
            "templates_array": self.templates_array.tolist(),
            "sparsity": sparsity,
            "num_units": self.num_units,
            "num_samples": self.num_samples,
            "num_channels": self.num_channels,
        }

    # Construct the object from a dictionary
    @classmethod
    def from_dict(cls, data):
        sparsity = ChannelSparsity.from_dict(data["sparsity"]) if data["sparsity"] is not None else None
        return cls(
            templates_array=np.array(data["templates_array"]),
            sparsity=sparsity,
        )

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))
