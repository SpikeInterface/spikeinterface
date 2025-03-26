import pytest

from pydantic import ValidationError
from pathlib import Path
import numpy as np

from spikeinterface.curation.curation_model import CurationModel

valid_split_1 = {"format_version": "1", "unit_ids": [1, 2, 3], "split_units": {1: [1, 2], 2: [2, 3], 3: [4, 5]}}


valid_split_2 = {
    "format_version": "1",
    "unit_ids": [1, 2, 3, 4],
    "split_units": {1: [[1, 2], [3, 4]], 2: [[2, 3], [4, 1]]},
}

invalid_split_1 = {
    "format_version": "1",
    "unit_ids": [1, 2, 3],
    "split_units": {1: [[1, 2], [2, 3]], 2: [2, 3], 3: [4, 5]},
}

invalid_split_2 = {"format_version": "1", "unit_ids": [1, 2, 3], "split_units": {4: [[1, 2], [2, 3]]}}


def test_unit_split():
    CurationModel(**valid_split_1)
    CurationModel(**valid_split_2)

    # shold raise error
    with pytest.raises(ValidationError):
        CurationModel(**invalid_split_1)
    with pytest.raises(ValidationError):
        CurationModel(**invalid_split_2)
