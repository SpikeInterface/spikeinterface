import pytest

from pydantic import ValidationError
import numpy as np

from spikeinterface.curation.curation_model import CurationModel, LabelDefinition


# Test data for format version
def test_format_version():
    # Valid format version
    CurationModel(format_version="1", unit_ids=[1, 2, 3])

    # Invalid format version
    with pytest.raises(ValidationError):
        CurationModel(format_version="2", unit_ids=[1, 2, 3])
    with pytest.raises(ValidationError):
        CurationModel(format_version="0", unit_ids=[1, 2, 3])


# Test data for label definitions
def test_label_definitions():
    valid_label_def = {
        "format_version": "1",
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
        "format_version": "1",
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
        "format_version": "1",
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
        "format_version": "1",
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
    valid_merge = {
        "format_version": "1",
        "unit_ids": [1, 2, 3, 4],
        "merge_unit_groups": [[1, 2], [3, 4]],
        "merge_new_unit_ids": [5, 6],
    }

    model = CurationModel(**valid_merge)
    assert len(model.merge_unit_groups) == 2
    assert len(model.merge_new_unit_ids) == 2

    # Test invalid merge group (single unit)
    invalid_merge_group = {"format_version": "1", "unit_ids": [1, 2, 3], "merge_unit_groups": [[1], [2, 3]]}
    with pytest.raises(ValidationError):
        CurationModel(**invalid_merge_group)

    # Test overlapping merge groups
    invalid_overlap = {"format_version": "1", "unit_ids": [1, 2, 3], "merge_unit_groups": [[1, 2], [2, 3]]}
    with pytest.raises(ValidationError):
        CurationModel(**invalid_overlap)

    # Test merge new unit IDs length mismatch
    invalid_new_ids = {
        "format_version": "1",
        "unit_ids": [1, 2, 3, 4],
        "merge_unit_groups": [[1, 2], [3, 4]],
        "merge_new_unit_ids": [5],  # Missing one ID
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_new_ids)


# Test removed units
def test_removed_units():
    valid_remove = {"format_version": "1", "unit_ids": [1, 2, 3], "removed_units": [2]}

    model = CurationModel(**valid_remove)
    assert len(model.removed_units) == 1

    # Test removing non-existent unit
    invalid_remove = {"format_version": "1", "unit_ids": [1, 2, 3], "removed_units": [4]}  # Non-existent unit
    with pytest.raises(ValidationError):
        CurationModel(**invalid_remove)

    # Test conflict between merge and remove
    invalid_merge_remove = {
        "format_version": "1",
        "unit_ids": [1, 2, 3],
        "merge_unit_groups": [[1, 2]],
        "removed_units": [1],  # Unit is both merged and removed
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_merge_remove)


# Test complete model with multiple operations
def test_complete_model():
    complete_model = {
        "format_version": "1",
        "unit_ids": [1, 2, 3, 4, 5],
        "label_definitions": {
            "quality": LabelDefinition(name="quality", label_options=["good", "noise"], exclusive=True),
            "tags": LabelDefinition(name="tags", label_options=["burst", "slow"], exclusive=False),
        },
        "manual_labels": [{"unit_id": 1, "labels": {"quality": ["good"], "tags": ["burst"]}}],
        "merge_unit_groups": [[2, 3]],
        "merge_new_unit_ids": [6],
        "split_units": {4: [[1, 2], [3, 4]]},
        "removed_units": [5],
    }

    model = CurationModel(**complete_model)
    assert model.format_version == "1"
    assert len(model.unit_ids) == 5
    assert len(model.label_definitions) == 2
    assert len(model.manual_labels) == 1
    assert len(model.merge_unit_groups) == 1
    assert len(model.merge_new_unit_ids) == 1
    assert len(model.split_units) == 1
    assert len(model.removed_units) == 1


# Test unit splitting functionality
def test_unit_split():
    # Test simple split (method 1)
    valid_simple_split = {
        "format_version": "1",
        "unit_ids": [1, 2, 3],
        "split_units": {
            1: [1, 2],  # Split unit 1 into two parts
            2: [2, 3],  # Split unit 2 into two parts
            3: [4, 5],  # Split unit 3 into two parts
        },
    }
    model = CurationModel(**valid_simple_split)
    assert len(model.split_units) == 3

    # Test complex split with multiple groups (method 2)
    valid_complex_split = {
        "format_version": "1",
        "unit_ids": [1, 2, 3, 4],
        "split_units": {
            1: [[1, 2], [3, 4]],  # Split unit 1 into two groups
            2: [[2, 3], [4, 1]],  # Split unit 2 into two groups
        },
    }
    model = CurationModel(**valid_complex_split)
    assert len(model.split_units) == 2

    # Test invalid mixing of methods
    invalid_mixed_methods = {
        "format_version": "1",
        "unit_ids": [1, 2, 3],
        "split_units": {
            1: [[1, 2], [2, 3]],  # Using method 2
            2: [2, 3],  # Using method 1
            3: [4, 5],  # Using method 1
        },
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_mixed_methods)

    # Test invalid unit ID
    invalid_unit_id = {
        "format_version": "1",
        "unit_ids": [1, 2, 3],
        "split_units": {4: [[1, 2], [2, 3]]},  # Unit 4 doesn't exist in unit_ids
    }
    with pytest.raises(ValidationError):
        CurationModel(**invalid_unit_id)
