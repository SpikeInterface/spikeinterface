import pytest

from pydantic import ValidationError
import numpy as np

from spikeinterface.curation.curation_model import CurationModel, SequentialCuration, LabelDefinition


# Test data for format version
def test_format_version():
    # Valid format version
    CurationModel(format_version="1", unit_ids=[1, 2, 3])

    # Invalid format version
    with pytest.raises(ValidationError):
        CurationModel(format_version="3", unit_ids=[1, 2, 3])
    with pytest.raises(ValidationError):
        CurationModel(format_version="0.1", unit_ids=[1, 2, 3])


# Test data for label definitions
def test_label_definitions():
    valid_label_def = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "label_definitions": {
            "quality": LabelDefinition(name="quality", label_options=["good", "noise"], exclusive=True),
            "tags": LabelDefinition(name="tags", label_options=["burst", "slow", "fast"], exclusive=False),
        },
    }

    model = CurationModel(**valid_label_def)
    assert "quality" in model.label_definitions
    assert model.label_definitions["quality"].name == "quality"
    assert model.label_definitions["quality"].exclusive is True

    # Test invalid label definition
    with pytest.raises(ValidationError):
        LabelDefinition(name="quality", label_options=[], exclusive=True)  # Empty options should be invalid


# Test manual labels
def test_manual_labels():
    valid_labels = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "label_definitions": {
            "quality": LabelDefinition(name="quality", label_options=["good", "noise"], exclusive=True),
            "tags": LabelDefinition(name="tags", label_options=["burst", "slow", "fast"], exclusive=False),
        },
        "manual_labels": [
            {"unit_id": 1, "labels": {"quality": ["good"], "tags": ["burst", "fast"]}},
            {"unit_id": 2, "labels": {"quality": ["noise"]}},
        ],
    }

    model = CurationModel(**valid_labels)
    assert len(model.manual_labels) == 2

    # Test invalid unit ID
    invalid_unit = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "label_definitions": {
            "quality": LabelDefinition(name="quality", label_options=["good", "noise"], exclusive=True)
        },
        "manual_labels": [{"unit_id": 4, "labels": {"quality": ["good"]}}],  # Non-existent unit
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_unit)

    # Test violation of exclusive label
    invalid_exclusive = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "label_definitions": {
            "quality": LabelDefinition(name="quality", label_options=["good", "noise"], exclusive=True)
        },
        "manual_labels": [
            {"unit_id": 1, "labels": {"quality": ["good", "noise"]}}  # Multiple values for exclusive label
        ],
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_exclusive)


# Test merge functionality
def test_merge_units():
    # Test list format
    valid_merge = {
        "format_version": "2",
        "unit_ids": [1, 2, 3, 4],
        "merges": [
            {"unit_ids": [1, 2], "new_unit_id": 5},
            {"unit_ids": [3, 4], "new_unit_id": 6},
        ],
    }

    model = CurationModel(**valid_merge)
    assert len(model.merges) == 2
    assert model.merges[0].new_unit_id == 5
    assert model.merges[1].new_unit_id == 6

    # Test dictionary format
    valid_merge_dict = {"format_version": "2", "unit_ids": [1, 2, 3, 4], "merges": {5: [1, 2], 6: [3, 4]}}

    model = CurationModel(**valid_merge_dict)
    assert len(model.merges) == 2
    merge_new_ids = {merge.new_unit_id for merge in model.merges}
    assert merge_new_ids == {5, 6}

    # Test list format
    valid_merge_list = {
        "format_version": "2",
        "unit_ids": [1, 2, 3, 4],
        "merges": [[1, 2], [3, 4]],  # Merge each pair into a new unit
    }
    model = CurationModel(**valid_merge_list)
    assert len(model.merges) == 2

    # Test invalid merge group (single unit)
    invalid_merge_group = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "merges": [{"unit_ids": [1], "new_unit_id": 4}],
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_merge_group)

    # Test overlapping merge groups
    invalid_overlap = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "merges": [
            {"unit_ids": [1, 2], "new_unit_id": 4},
            {"unit_ids": [2, 3], "new_unit_id": 5},
        ],
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_overlap)


# Test split functionality
def test_split_units():
    # Test indices mode with list format
    valid_split_indices = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "splits": [
            {
                "unit_id": 1,
                "mode": "indices",
                "indices": [[0, 1, 2], [3, 4, 5]],
                "new_unit_ids": [4, 5],
            }
        ],
    }

    model = CurationModel(**valid_split_indices)
    assert len(model.splits) == 1
    assert model.splits[0].mode == "indices"
    assert len(model.splits[0].indices) == 2

    # Test labels mode with list format
    valid_split_labels = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "splits": [{"unit_id": 1, "mode": "labels", "labels": [0, 0, 1, 1, 0, 2], "new_unit_ids": [4, 5, 6]}],
    }

    model = CurationModel(**valid_split_labels)
    assert len(model.splits) == 1
    assert model.splits[0].mode == "labels"
    assert len(set(model.splits[0].labels)) == 3

    # Test dictionary format with indices
    valid_split_dict = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "splits": {
            1: [[0, 1, 2], [3, 4, 5]],  # Split unit 1 into two parts
            2: [[0, 1], [2, 3], [4, 5]],  # Split unit 2 into three parts
        },
    }

    model = CurationModel(**valid_split_dict)
    assert len(model.splits) == 2
    assert all(split.mode == "indices" for split in model.splits)

    # Test invalid unit ID
    invalid_unit_id = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "splits": [{"unit_id": 4, "mode": "indices", "indices": [[0, 1], [2, 3]]}],  # Non-existent unit
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_unit_id)

    # Test invalid new unit IDs count for indices mode
    invalid_new_ids = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "splits": [
            {
                "unit_id": 1,
                "mode": "indices",
                "indices": [[0, 1], [2, 3]],
                "new_unit_ids": [4],  # Should have 2 new IDs for 2 splits
            }
        ],
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_new_ids)


# Test removed units
def test_removed_units():
    valid_remove = {"format_version": "2", "unit_ids": [1, 2, 3], "removed": [2]}

    model = CurationModel(**valid_remove)
    assert len(model.removed) == 1

    # Test removing non-existent unit
    invalid_remove = {"format_version": "2", "unit_ids": [1, 2, 3], "removed": [4]}  # Non-existent unit
    with pytest.raises(ValidationError):
        CurationModel(**invalid_remove)

    # Test conflict between merge and remove
    invalid_merge_remove = {
        "format_version": "2",
        "unit_ids": [1, 2, 3],
        "merges": [{"unit_ids": [1, 2], "new_unit_id": 4}],
        "removed": [1],  # Unit is both merged and removed
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_merge_remove)


# Test complete model with multiple operations
def test_complete_model():
    complete_model = {
        "format_version": "2",
        "unit_ids": [1, 2, 3, 4, 5],
        "label_definitions": {
            "quality": LabelDefinition(name="quality", label_options=["good", "noise"], exclusive=True),
            "tags": LabelDefinition(name="tags", label_options=["burst", "slow"], exclusive=False),
        },
        "manual_labels": [{"unit_id": 1, "labels": {"quality": ["good"], "tags": ["burst"]}}],
        "merges": [{"unit_ids": [2, 3], "new_unit_id": 6}],
        "splits": [{"unit_id": 4, "mode": "indices", "indices": [[0, 1], [2, 3]], "new_unit_ids": [7, 8]}],
        "removed": [5],
    }

    model = CurationModel(**complete_model)
    assert model.format_version == "2"
    assert len(model.unit_ids) == 5
    assert len(model.label_definitions) == 2
    assert len(model.manual_labels) == 1
    assert len(model.merges) == 1
    assert len(model.splits) == 1
    assert len(model.removed) == 1

    # Test dictionary format for complete model
    complete_model_dict = {
        "format_version": "2",
        "unit_ids": [1, 2, 3, 4, 5],
        "label_definitions": {
            "quality": LabelDefinition(name="quality", label_options=["good", "noise"], exclusive=True),
            "tags": LabelDefinition(name="tags", label_options=["burst", "slow"], exclusive=False),
        },
        "manual_labels": [{"unit_id": 1, "labels": {"quality": ["good"], "tags": ["burst"]}}],
        "merges": {6: [2, 3]},
        "splits": {4: [[0, 1], [2, 3]]},
        "removed": [5],
    }

    model = CurationModel(**complete_model_dict)
    assert model.format_version == "2"
    assert len(model.unit_ids) == 5
    assert len(model.label_definitions) == 2
    assert len(model.manual_labels) == 1
    assert len(model.merges) == 1
    assert len(model.splits) == 1
    assert len(model.removed) == 1


def test_sequential_curation():
    sequential_curation_steps_valid = [
        {"format_version": "2", "unit_ids": [1, 2, 3, 4], "merges": [{"unit_ids": [1, 2], "new_unit_id": 22}]},
        {
            "format_version": "2",
            "unit_ids": [3, 4, 22],
            "splits": [
                {"unit_id": 22, "mode": "indices", "indices": [[0, 1, 2], [3, 4, 5]], "new_unit_ids": [222, 223]}
            ],
        },
        {"format_version": "2", "unit_ids": [3, 4, 222, 223], "removed": [223]},
    ]

    # this is valid
    SequentialCuration(curation_steps=sequential_curation_steps_valid)

    sequential_curation_steps_no_ids = sequential_curation_steps_valid.copy()
    # remove new_unit_id in merge step
    sequential_curation_steps_no_ids[0]["merges"][0]["new_unit_id"] = None

    with pytest.raises(ValidationError):
        SequentialCuration(curation_steps=sequential_curation_steps_no_ids)

    sequential_curation_steps_invalid = sequential_curation_steps_valid.copy()
    # invalid unit_ids in last step
    sequential_curation_steps_invalid[2]["unit_ids"] = [3, 4, 222, 224]  # 224 should be 223
    with pytest.raises(ValidationError):
        SequentialCuration(curation_steps=sequential_curation_steps_invalid)
