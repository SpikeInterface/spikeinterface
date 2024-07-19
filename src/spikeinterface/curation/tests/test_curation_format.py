import pytest

from pathlib import Path
import json
import numpy as np

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer

from spikeinterface.curation.curation_format import (
    validate_curation_dict,
    convert_from_sortingview_curation_format_v0,
    curation_label_to_vectors,
    curation_label_to_dataframe,
    apply_curation,
)


"""example = {
    'unit_ids': List[str, int],
    'label_definitions': {
        'category_key1':
        {
         'label_options': List[str],
         'exclusive': bool}
    },
    'manual_labels': [
        {'unit_id': str or int,
         category_key1': List[str],
         }
    ],
    'merge_unit_groups': List[List[unit_ids]],  # one cell goes into at most one list
    'removed_units': List[unit_ids]  # Can not be  in the merged_units
}
"""


curation_ids_int = {
    "format_version": "1",
    "unit_ids": [1, 2, 3, 6, 10, 14, 20, 31, 42],
    "label_definitions": {
        "quality": {"label_options": ["good", "noise", "MUA", "artifact"], "exclusive": True},
        "putative_type": {
            "label_options": ["excitatory", "inhibitory", "pyramidal", "mitral"],
            "exclusive": False,
        },
    },
    "manual_labels": [
        {"unit_id": 1, "quality": ["good"]},
        {
            "unit_id": 2,
            "quality": [
                "noise",
            ],
            "putative_type": ["excitatory", "pyramidal"],
        },
        {"unit_id": 3, "putative_type": ["inhibitory"]},
    ],
    "merge_unit_groups": [[3, 6], [10, 14, 20]],  # one cell goes into at most one list
    "removed_units": [31, 42],  # Can not be  in the merged_units
}

curation_ids_str = {
    "format_version": "1",
    "unit_ids": ["u1", "u2", "u3", "u6", "u10", "u14", "u20", "u31", "u42"],
    "label_definitions": {
        "quality": {"label_options": ["good", "noise", "MUA", "artifact"], "exclusive": True},
        "putative_type": {
            "label_options": ["excitatory", "inhibitory", "pyramidal", "mitral"],
            "exclusive": False,
        },
    },
    "manual_labels": [
        {"unit_id": "u1", "quality": ["good"]},
        {
            "unit_id": "u2",
            "quality": [
                "noise",
            ],
            "putative_type": ["excitatory", "pyramidal"],
        },
        {"unit_id": "u3", "putative_type": ["inhibitory"]},
    ],
    "merge_unit_groups": [["u3", "u6"], ["u10", "u14", "u20"]],  # one cell goes into at most one list
    "removed_units": ["u31", "u42"],  # Can not be  in the merged_units
}

# This is a failure example with duplicated merge
duplicate_merge = curation_ids_int.copy()
duplicate_merge["merge_unit_groups"] = [[3, 6, 10], [10, 14, 20]]


# This is a failure example with unit 3 both in removed and merged
merged_and_removed = curation_ids_int.copy()
merged_and_removed["merge_unit_groups"] = [[3, 6], [10, 14, 20]]
merged_and_removed["removed_units"] = [3, 31, 42]

# this is a failure because unit 99 is not in the initial list
unknown_merged_unit = curation_ids_int.copy()
unknown_merged_unit["merge_unit_groups"] = [[3, 6, 99], [10, 14, 20]]

# this is a failure because unit 99 is not in the initial list
unknown_removed_unit = curation_ids_int.copy()
unknown_removed_unit["removed_units"] = [31, 42, 99]


def test_curation_format_validation():
    validate_curation_dict(curation_ids_int)
    validate_curation_dict(curation_ids_str)

    with pytest.raises(ValueError):
        # Raised because duplicated merged units
        validate_curation_dict(duplicate_merge)
    with pytest.raises(ValueError):
        # Raised because some units belong to merged and removed unit groups
        validate_curation_dict(merged_and_removed)
    with pytest.raises(ValueError):
        # Some merged units are not in the unit list
        validate_curation_dict(unknown_merged_unit)
    with pytest.raises(ValueError):
        # Raise because some removed units are not in the unit list
        validate_curation_dict(unknown_removed_unit)


def test_to_from_json():

    json.loads(json.dumps(curation_ids_int, indent=4))
    json.loads(json.dumps(curation_ids_str, indent=4))


def test_convert_from_sortingview_curation_format_v0():

    parent_folder = Path(__file__).parent
    for filename in (
        "sv-sorting-curation.json",
        "sv-sorting-curation-int.json",
        "sv-sorting-curation-str.json",
        "sv-sorting-curation-false-positive.json",
    ):

        json_file = parent_folder / filename
        with open(json_file, "r") as f:
            curation_v0 = json.load(f)
            # print(curation_v0)
            curation_v1 = convert_from_sortingview_curation_format_v0(curation_v0)
            # print(curation_v1)
            validate_curation_dict(curation_v1)


def test_curation_label_to_vectors():

    labels = curation_label_to_vectors(curation_ids_int)
    assert "quality" in labels
    assert "excitatory" in labels
    print(labels)

    labels = curation_label_to_vectors(curation_ids_str)
    print(labels)


def test_curation_label_to_dataframe():

    df = curation_label_to_dataframe(curation_ids_int)
    assert "quality" in df.columns
    assert "excitatory" in df.columns
    print(df)

    df = curation_label_to_dataframe(curation_ids_str)
    # print(df)


def test_apply_curation():
    recording, sorting = generate_ground_truth_recording(durations=[10.0], num_units=9, seed=2205)
    sorting._main_ids = np.array([1, 2, 3, 6, 10, 14, 20, 31, 42])
    analyzer = create_sorting_analyzer(sorting, recording, sparse=False)

    sorting_curated = apply_curation(sorting, curation_ids_int)
    assert sorting_curated.get_property("quality", ids=[1])[0] == "good"
    assert sorting_curated.get_property("quality", ids=[2])[0] == "noise"
    assert sorting_curated.get_property("excitatory", ids=[2])[0]

    analyzer_curated = apply_curation(analyzer, curation_ids_int)
    assert "quality" in analyzer_curated.sorting.get_property_keys()


if __name__ == "__main__":
    # test_curation_format_validation()
    # test_to_from_json()
    # test_convert_from_sortingview_curation_format_v0()
    # test_curation_label_to_vectors()
    # test_curation_label_to_dataframe()

    test_apply_curation()
