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
            'merge_unit_group': List[unit_ids],
            'merge_new_unit_id': int | str (optional)
        }
    ],
    'splits': [
        {
            'unit_id': int | str
            'split_mode': 'indices' or 'labels',
            'split_indices': List[List[int]],
            'split_labels': List[int]],
            'split_new_unit_ids': List[int | str]
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
    "merges": [{"merge_unit_group": [3, 6]}, {"merge_unit_group": [10, 14, 20]}],
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
    "merges": [{"merge_unit_group": ["u3", "u6"]}, {"merge_unit_group": ["u10", "u14", "u20"]}],
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
    "splits": [{"unit_id": 2, "split_mode": "indices", "split_indices": [[0, 1, 2], [3, 4, 5]]}],
}

# Test dictionary format for splits
curation_with_splits_dict = {**curation_ids_int, "splits": {2: [[0, 1, 2], [3, 4, 5]]}}

# This is a failure example with duplicated merge
duplicate_merge = {**curation_ids_int, "merges": [{"merge_unit_group": [3, 6, 10]}, {"merge_unit_group": [10, 14, 20]}]}

# This is a failure example with unit 3 both in removed and merged
merged_and_removed = {
    **curation_ids_int,
    "merges": [{"merge_unit_group": [3, 6]}, {"merge_unit_group": [10, 14, 20]}],
    "removed": [3, 31, 42],
}

# This is a failure because unit 99 is not in the initial list
unknown_merged_unit = {
    **curation_ids_int,
    "merges": [{"merge_unit_group": [3, 6, 99]}, {"merge_unit_group": [10, 14, 20]}],
}

# This is a failure because unit 99 is not in the initial list
unknown_removed_unit = {**curation_ids_int, "removed": [31, 42, 99]}


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

    curation_with_splits_multi_segment["splits"][0]["split_indices"] = split_indices

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
                "split_mode": "labels",
                "split_labels": split_labels.tolist(),
                "split_new_unit_ids": [43, 44, 45],
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


if __name__ == "__main__":
    test_curation_format_validation()
    test_to_from_json()
    test_convert_from_sortingview_curation_format_v0()
    test_curation_label_to_vectors()
    test_curation_label_to_dataframe()
    test_apply_curation()
    test_apply_curation_with_split_multi_segment()
    test_apply_curation_splits_with_mask()
