import pytest

from pydantic import BaseModel, ValidationError, field_validator


from pathlib import Path
import json
import numpy as np

from spikeinterface.curation.curation_model import CurationModel

values_1 = { "format_version": "1",
    "unit_ids": [1, 2, 3],
    "split_units": {1: [1, 2], 2: [2, 3],3: [4,5]}
}


values_2 = { "format_version": "1",
    "unit_ids": [1, 2, 3, 4],
    "split_units": {
        1: [[1, 2], [3, 4]],  
        2: [[2, 3], [4, 1]]
    }
}

curation_model1 = CurationModel(**values_1)
curation_model = CurationModel(**values_2)

    