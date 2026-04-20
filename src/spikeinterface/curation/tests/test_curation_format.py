import pytest

from pathlib import Path
import json
import numpy as np

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer

from spikeinterface.curation.curation_format import (
    validate_curation_dict,
    curation_label_to_vectors,
    curation_label_to_dataframe,
    apply_curation,
    load_curation,
)

"""
v1 = {
    'format_version': '1',
    'unit_ids': List[int | str],
    'label_definitions': {
        'category_key1':
        {
         'label_options': List[str],
         'exclusive': bool}
    },
    'manual_labels': [
        {
            'unit_id': str or int,
            'category_key1': List[str],
        }
    ],
    'removed_units': List[int | str]  # Can not be  in the merged_units
    'merge_unit_groups': List[List[int | str]],  # one cell goes into at most one list
}

v2 = {
    'format_version': '2',
    'unit_ids': List[int | int],
    'label_definitions': {
        'category_key1':
        {
         'label_options': List[str],
         'exclusive': bool}
    },
    'manual_labels': [
        {
            'unit_id': str | int,
            'category_key1': List[str],
         }
    ],
    'removed': List[unit_ids], # Can not be  in the merged_units
    'merges': [
        {
            'unit_ids': List[unit_ids],
            'new_unit_id': int | str (optional)
        }
    ],
    'splits': [
        {
            'unit_id': int | str
            'mode': 'indices' or 'labels',
            'indices': List[List[int]],
            'labels': List[int]],
            'new_unit_ids': List[int | str]
        }
    ]

sortingview_curation = {
    'mergeGroups': List[List[int | str]],
    'labelsByUnit': {
        'unit_id': List[str]
    }

"""

curation_ids_int = {
    "format_version": "2",
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
            "quality": ["noise"],
            "putative_type": ["excitatory", "pyramidal"],
        },
        {"unit_id": 3, "putative_type": ["inhibitory"]},
    ],
    "merges": [{"unit_ids": [3, 6]}, {"unit_ids": [10, 14, 20]}],
    "splits": [],
    "removed": [31, 42],
}

# Test dictionary format for merges
curation_ids_int_dict = {**curation_ids_int, "merges": {50: [3, 6], 51: [10, 14, 20]}}

curation_ids_str = {
    "format_version": "2",
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
            "quality": ["noise"],
            "putative_type": ["excitatory", "pyramidal"],
        },
        {"unit_id": "u3", "putative_type": ["inhibitory"]},
    ],
    "merges": [{"unit_ids": ["u3", "u6"]}, {"unit_ids": ["u10", "u14", "u20"]}],
    "splits": [],
    "removed": ["u31", "u42"],
}

# Test dictionary format for merges with string IDs
curation_ids_str_dict = {**curation_ids_str, "merges": {"u50": ["u3", "u6"], "u51": ["u10", "u14", "u20"]}}

# This is a failure example with duplicated merge
duplicate_merge = curation_ids_int.copy()
duplicate_merge["merge_unit_groups"] = [[3, 6, 10], [10, 14, 20]]

# Test with splits
curation_with_splits = {
    "format_version": "2",
    "unit_ids": [1, 2, 3, 6, 10, 14, 20, 31, 42],
    "label_definitions": {
        "quality": {"label_options": ["good", "noise", "MUA", "artifact"], "exclusive": True},
        "putative_type": {
            "label_options": ["excitatory", "inhibitory", "pyramidal", "mitral"],
            "exclusive": False,
        },
    },
    "manual_labels": [
        {"unit_id": 2, "quality": ["good"], "putative_type": ["excitatory", "pyramidal"]},
    ],
    "splits": [{"unit_id": 2, "mode": "indices", "indices": [[0, 1, 2], [3, 4, 5]]}],
}

# Test dictionary format for splits
curation_with_splits_dict = {**curation_ids_int, "splits": {2: [[0, 1, 2], [3, 4, 5]]}}

# This is a failure example with duplicated merge
duplicate_merge = {**curation_ids_int, "merges": [{"unit_ids": [3, 6, 10]}, {"unit_ids": [10, 14, 20]}]}

# This is a failure example with unit 3 both in removed and merged
merged_and_removed = {
    **curation_ids_int,
    "merges": [{"unit_ids": [3, 6]}, {"unit_ids": [10, 14, 20]}],
    "removed": [3, 31, 42],
}

# This is a failure because unit 99 is not in the initial list
unknown_merged_unit = {
    **curation_ids_int,
    "merges": [{"unit_ids": [3, 6, 99]}, {"unit_ids": [10, 14, 20]}],
}

# This is a failure because unit 99 is not in the initial list
unknown_removed_unit = {**curation_ids_int, "removed": [31, 42, 99]}

# Sequential curation test data
sequential_curation = [
    {
        "format_version": "2",
        "unit_ids": [1, 2, 3, 4, 5],
        "merges": [{"unit_ids": [3, 4], "new_unit_id": 34}],
    },
    {
        "format_version": "2",
        "unit_ids": [1, 2, 34, 5],
        "splits": [{"unit_id": 34, "mode": "indices", "indices": [[0, 1, 2, 3]], "new_unit_ids": [340, 341]}],
    },
    {
        "format_version": "2",
        "unit_ids": [1, 2, 340, 341, 5],
        "removed": [2, 5],
        "merges": [{"unit_ids": [1, 340], "new_unit_id": 100}],
        "splits": [{"unit_id": 341, "mode": "indices", "indices": [[0, 1, 2]], "new_unit_ids": [3410, 3411]}],
    },
]


def test_curation_format_validation():
    # Test basic formats
    print(curation_ids_int)
    validate_curation_dict(curation_ids_int)
    print(curation_ids_int)
    validate_curation_dict(curation_ids_str)

    # Test dictionary formats
    validate_curation_dict(curation_ids_int_dict)
    validate_curation_dict(curation_ids_str_dict)

    # Test splits
    validate_curation_dict(curation_with_splits)
    validate_curation_dict(curation_with_splits_dict)

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
    json.loads(json.dumps(curation_ids_int_dict, indent=4))
    json.loads(json.dumps(curation_with_splits, indent=4))


def test_to_from_json_file(tmp_path):

    for curation_index, curation_dict in enumerate(
        [curation_ids_int, curation_ids_str, curation_ids_int_dict, curation_with_splits]
    ):

        temp_filepath = tmp_path / f"data_{curation_index}.json"

        with open(temp_filepath, "w") as f:
            json.dump(curation_dict, f)

        curation = load_curation(temp_filepath)
        curation.validate_curation_dict()


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
            validate_curation_dict(curation_v0)


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
    print(df)


def test_apply_curation():
    recording, sorting = generate_ground_truth_recording(durations=[10.0], num_units=9, seed=2205)
    sorting = sorting.rename_units([1, 2, 3, 6, 10, 14, 20, 31, 42])
    analyzer = create_sorting_analyzer(sorting, recording, sparse=False)

    # Test with list format
    sorting_curated = apply_curation(sorting, curation_ids_int)
    assert sorting_curated.get_property("quality", ids=[1])[0] == "good"
    assert sorting_curated.get_property("quality", ids=[2])[0] == "noise"
    assert sorting_curated.get_property("excitatory", ids=[2])[0]

    # Test with dictionary format
    sorting_curated = apply_curation(sorting, curation_ids_int_dict)
    assert sorting_curated.get_property("quality", ids=[1])[0] == "good"
    assert sorting_curated.get_property("quality", ids=[2])[0] == "noise"
    assert sorting_curated.get_property("excitatory", ids=[2])[0]

    # Test analyzer
    analyzer_curated = apply_curation(analyzer, curation_ids_int)
    assert "quality" in analyzer_curated.sorting.get_property_keys()


def test_apply_curation_with_split():
    recording, sorting = generate_ground_truth_recording(durations=[10.0], num_units=9, seed=2205)
    sorting = sorting.rename_units(np.array([1, 2, 3, 6, 10, 14, 20, 31, 42]))
    analyzer = create_sorting_analyzer(sorting, recording, sparse=False)

    sorting_curated = apply_curation(sorting, curation_with_splits)
    # the split indices are not complete, so an extra unit is added
    assert len(sorting_curated.unit_ids) == len(sorting.unit_ids) + 2

    assert 2 not in sorting_curated.unit_ids
    split_unit_ids = [43, 44, 45]
    for unit_id in split_unit_ids:
        assert unit_id in sorting_curated.unit_ids
        assert sorting_curated.get_property("quality", ids=[unit_id])[0] == "good"
        assert sorting_curated.get_property("excitatory", ids=[unit_id])[0]
        assert sorting_curated.get_property("pyramidal", ids=[unit_id])[0]

    analyzer_curated = apply_curation(analyzer, curation_with_splits)
    assert len(analyzer_curated.sorting.unit_ids) == len(analyzer.sorting.unit_ids) + 2

    assert 2 not in analyzer_curated.unit_ids
    for unit_id in split_unit_ids:
        assert unit_id in analyzer_curated.unit_ids
        assert analyzer_curated.sorting.get_property("quality", ids=[unit_id])[0] == "good"
        assert analyzer_curated.sorting.get_property("excitatory", ids=[unit_id])[0]
        assert analyzer_curated.sorting.get_property("pyramidal", ids=[unit_id])[0]


def test_apply_curation_with_split_multi_segment():
    recording, sorting = generate_ground_truth_recording(durations=[10.0, 10.0], num_units=9, seed=2205)
    sorting = sorting.rename_units(np.array([1, 2, 3, 6, 10, 14, 20, 31, 42]))
    analyzer = create_sorting_analyzer(sorting, recording, sparse=False)
    num_segments = sorting.get_num_segments()

    curation_with_splits_multi_segment = curation_with_splits.copy()

    # we make a split so that each subsplit will have all spikes from different segments
    split_unit_id = curation_with_splits_multi_segment["splits"][0]["unit_id"]
    sv = sorting.to_spike_vector()
    unit_index = sorting.id_to_index(split_unit_id)
    spikes_from_split_unit = sv[sv["unit_index"] == unit_index]

    split_indices = []
    cum_spikes = 0
    for segment_index in range(num_segments):
        spikes_in_segment = spikes_from_split_unit[spikes_from_split_unit["segment_index"] == segment_index]
        split_indices.append(np.arange(0, len(spikes_in_segment)) + cum_spikes)
        cum_spikes += len(spikes_in_segment)

    curation_with_splits_multi_segment["splits"][0]["indices"] = split_indices

    sorting_curated = apply_curation(sorting, curation_with_splits_multi_segment)

    assert len(sorting_curated.unit_ids) == len(sorting.unit_ids) + 1
    assert 2 not in sorting_curated.unit_ids
    assert 43 in sorting_curated.unit_ids
    assert 44 in sorting_curated.unit_ids

    # check that spike trains are correctly split across segments
    for seg_index in range(num_segments):
        st_43 = sorting_curated.get_unit_spike_train(43, segment_index=seg_index)
        st_44 = sorting_curated.get_unit_spike_train(44, segment_index=seg_index)
        if seg_index == 0:
            assert len(st_43) > 0
            assert len(st_44) == 0
        else:
            assert len(st_43) == 0
            assert len(st_44) > 0


def test_apply_curation_splits_with_mask():
    recording, sorting = generate_ground_truth_recording(durations=[10.0], num_units=9, seed=2205)
    sorting = sorting.rename_units(np.array([1, 2, 3, 6, 10, 14, 20, 31, 42]))
    analyzer = create_sorting_analyzer(sorting, recording, sparse=False)

    # Get number of spikes for unit 2
    num_spikes = sorting.count_num_spikes_per_unit()[2]

    # Create split labels that assign spikes to 3 different clusters
    split_labels = np.zeros(num_spikes, dtype=int)
    split_labels[: num_spikes // 3] = 0  # First third to cluster 0
    split_labels[num_spikes // 3 : 2 * num_spikes // 3] = 1  # Second third to cluster 1
    split_labels[2 * num_spikes // 3 :] = 2  # Last third to cluster 2

    curation_with_mask_split = {
        "format_version": "2",
        "unit_ids": [1, 2, 3, 6, 10, 14, 20, 31, 42],
        "label_definitions": {
            "quality": {"label_options": ["good", "noise", "MUA", "artifact"], "exclusive": True},
            "putative_type": {
                "label_options": ["excitatory", "inhibitory", "pyramidal", "mitral"],
                "exclusive": False,
            },
        },
        "manual_labels": [
            {"unit_id": 2, "quality": ["good"], "putative_type": ["excitatory", "pyramidal"]},
        ],
        "splits": [
            {
                "unit_id": 2,
                "mode": "labels",
                "labels": split_labels.tolist(),
                "new_unit_ids": [43, 44, 45],
            }
        ],
    }

    sorting_curated = apply_curation(sorting, curation_with_mask_split)

    # Check results
    assert len(sorting_curated.unit_ids) == len(sorting.unit_ids) + 2  # Original units - 1 (split) + 3 (new)
    assert 2 not in sorting_curated.unit_ids  # Original unit should be removed

    # Check new split units
    split_unit_ids = [43, 44, 45]
    for unit_id in split_unit_ids:
        assert unit_id in sorting_curated.unit_ids
        # Check properties are propagated
        assert sorting_curated.get_property("quality", ids=[unit_id])[0] == "good"
        assert sorting_curated.get_property("excitatory", ids=[unit_id])[0]
        assert sorting_curated.get_property("pyramidal", ids=[unit_id])[0]

    # Check analyzer
    analyzer_curated = apply_curation(analyzer, curation_with_mask_split)
    assert len(analyzer_curated.sorting.unit_ids) == len(analyzer.sorting.unit_ids) + 2

    # Verify split sizes
    spike_counts = analyzer_curated.sorting.count_num_spikes_per_unit()
    assert spike_counts[43] == num_spikes // 3  # First third
    assert spike_counts[44] == num_spikes // 3  # Second third
    assert spike_counts[45] == num_spikes - 2 * (num_spikes // 3)  # Remainder


def test_apply_sequential_curation():
    recording, sorting = generate_ground_truth_recording(durations=[10.0], num_units=5, seed=2205)
    sorting = sorting.rename_units([1, 2, 3, 4, 5])
    analyzer = create_sorting_analyzer(sorting, recording, sparse=False)

    # sequential curation steps:
    # 1. merge 3 and 4 -> 34
    # 2. split 34 -> 340, 341
    # 3. remove 2, 5; merge 1 and 340 -> 100; split 341 -> 3410, 3411
    analyzer_curated = apply_curation(analyzer, sequential_curation, verbose=True)
    # initial -1(merge) +1(split) -2(remove) -1(merge) +1(split)
    num_final_units = analyzer.get_num_units() - 1 + 1 - 2 - 1 + 1
    assert analyzer_curated.get_num_units() == num_final_units

    # check final unit ids
    final_unit_ids = analyzer_curated.sorting.unit_ids
    expected_final_unit_ids = [100, 3410, 3411]
    assert set(final_unit_ids) == set(expected_final_unit_ids)


if __name__ == "__main__":
    test_curation_format_validation()
    test_to_from_json()
    test_convert_from_sortingview_curation_format_v0()
    test_curation_label_to_vectors()
    test_curation_label_to_dataframe()
    test_apply_curation()
    test_apply_curation_with_split_multi_segment()
    test_apply_curation_splits_with_mask()
