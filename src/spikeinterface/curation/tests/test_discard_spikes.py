import numpy as np
from spikeinterface.core import NumpySorting
from spikeinterface.curation import apply_curation
from numpy.random import default_rng


def test_discard_and_split():
    rng = default_rng()
    spike_indices = [
        {
            0: np.sort(rng.choice(100, size=20, replace=False)),
            1: np.arange(17),
            2: np.arange(17) + 5,
            4: np.concatenate([np.arange(10), np.arange(20, 30)]),
            5: np.arange(9),
        },
        {0: np.arange(15), 1: np.arange(17), 2: np.arange(40, 140), 4: np.arange(40, 140), 5: np.arange(40, 140)},
    ]
    original_sort = NumpySorting.from_unit_dict(spike_indices, sampling_frequency=1000)  # to have 1 sample=1ms

    discard_spikes = [4, 11, 13, 22, 30]
    discard_spikes_segs = [[4, 11, 13], np.array([22, 30]) - len(spike_indices[0][0])]

    discard_spikes_curation = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "discard_spikes": [
            {
                "unit_id": 0,
                "indices": discard_spikes,
            }
        ],
    }

    curated_sort = apply_curation(original_sort, discard_spikes_curation)

    for segment_index in [0, 1]:

        original_spike_train = original_sort.get_unit_spike_train(unit_id=0, segment_index=segment_index)
        curated_spike_train = curated_sort.get_unit_spike_train(unit_id=0, segment_index=segment_index)

        discard_spike_times = set(original_spike_train).difference(set(curated_spike_train))
        spike_times_of_discarded_spikes = original_spike_train[discard_spikes_segs[segment_index]]

        assert set(discard_spike_times) == set(spike_times_of_discarded_spikes)

    split_indices = [2, 4, 5, 10, 13, 18, 22, 31, 33]
    split_spikes_segs_50 = [
        [2, 4, 5, 10, 13, 18],
        [22 - len(spike_indices[0][0]), 31 - len(spike_indices[0][0]), 33 - len(spike_indices[0][0])],
    ]
    indices_in_51_seg_2 = np.array([20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34])
    split_spikes_segs_51 = [
        [0, 1, 3, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19],
        indices_in_51_seg_2 - len(spike_indices[0][0]),
    ]
    split_spikes_segs = [split_spikes_segs_50, split_spikes_segs_51]

    discard_spikes_curation_with_split = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "discard_spikes": [
            {
                "unit_id": 0,
                "indices": discard_spikes,
            }
        ],
        "splits": [{"unit_id": 0, "mode": "indices", "indices": [split_indices], "new_unit_ids": [50, 51]}],
    }

    # 4 and 13 should be discarded
    should_be_discarded_spikes = [[[4, 13], [22 - len(spike_indices[0][0])]], [[11], [30 - len(spike_indices[0][0])]]]

    curated_sort_with_split = apply_curation(original_sort, discard_spikes_curation_with_split)

    for new_unit_id, discarded_spikes_each_unit, split_spikes_seg in zip(
        [50, 51], should_be_discarded_spikes, split_spikes_segs
    ):
        for segment_index in [0, 1]:

            original_spike_train = original_sort.get_unit_spike_train(unit_id=0, segment_index=segment_index)
            curated_spike_train = curated_sort_with_split.get_unit_spike_train(
                unit_id=new_unit_id, segment_index=segment_index
            )

            discard_spike_times = set(original_spike_train[split_spikes_seg[segment_index]]).difference(
                set(curated_spike_train)
            )

            spike_times_of_discarded_spikes = original_spike_train[discarded_spikes_each_unit[segment_index]]

            assert set(discard_spike_times) == set(spike_times_of_discarded_spikes)

    # test with "append" unit_id strategy

    split_indices = [2, 4, 5, 10, 13, 18, 22, 31, 33]
    split_spikes_segs_50 = [[2, 4, 5, 10, 13, 18], np.array([22, 31, 33]) - len(spike_indices[0][0])]
    indices_in_51_seg_2 = np.array([20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34])
    split_spikes_segs_51 = [
        [0, 1, 3, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19],
        indices_in_51_seg_2 - len(spike_indices[0][0]),
    ]
    split_spikes_segs = [split_spikes_segs_50, split_spikes_segs_51]

    discard_spikes_curation_with_split_append = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "discard_spikes": [
            {
                "unit_id": 0,
                "indices": discard_spikes,
            }
        ],
        "splits": [{"unit_id": 0, "mode": "indices", "indices": [split_indices]}],
    }

    # since we're using append, the new units will have id 6 and 7
    curated_sort_with_split_append = apply_curation(original_sort, discard_spikes_curation_with_split_append)

    print(f"{curated_sort_with_split_append.unit_ids=}", flush=True)

    for new_unit_id, discarded_spikes_each_unit, split_spikes_seg in zip(
        [6, 7], should_be_discarded_spikes, split_spikes_segs
    ):
        for segment_index in [0, 1]:

            original_spike_train = original_sort.get_unit_spike_train(unit_id=0, segment_index=segment_index)
            curated_spike_train = curated_sort_with_split_append.get_unit_spike_train(
                unit_id=new_unit_id, segment_index=segment_index
            )

            discard_spike_times = set(original_spike_train[split_spikes_seg[segment_index]]).difference(
                set(curated_spike_train)
            )

            spike_times_of_discarded_spikes = original_spike_train[discarded_spikes_each_unit[segment_index]]

            assert set(discard_spike_times) == set(spike_times_of_discarded_spikes)


def test_discard_and_split_string_ids():
    rng = default_rng()
    spike_indices = [
        {
            "0": np.sort(rng.choice(100, size=20, replace=False)),
            "1": np.arange(17),
            "2": np.arange(17) + 5,
            "4": np.concatenate([np.arange(10), np.arange(20, 30)]),
            "5": np.arange(9),
        },
        {
            "0": np.arange(15),
            "1": np.arange(17),
            "2": np.arange(40, 140),
            "4": np.arange(40, 140),
            "5": np.arange(40, 140),
        },
    ]
    original_sort = NumpySorting.from_unit_dict(spike_indices, sampling_frequency=1000)  # to have 1 sample=1ms

    discard_spikes = [4, 11, 13, 22, 30]
    discard_spikes_segs = [[4, 11, 13], np.array([22, 30]) - len(spike_indices[0]["0"])]

    discard_spikes_curation = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "discard_spikes": [
            {
                "unit_id": "0",
                "indices": discard_spikes,
            }
        ],
    }

    curated_sort = apply_curation(original_sort, discard_spikes_curation)

    for segment_index in [0, 1]:

        original_spike_train = original_sort.get_unit_spike_train(unit_id="0", segment_index=segment_index)
        curated_spike_train = curated_sort.get_unit_spike_train(unit_id="0", segment_index=segment_index)

        discard_spike_times = set(original_spike_train).difference(set(curated_spike_train))
        spike_times_of_discarded_spikes = original_spike_train[discard_spikes_segs[segment_index]]

        assert set(discard_spike_times) == set(spike_times_of_discarded_spikes)

    split_indices = [2, 4, 5, 10, 13, 18, 22, 31, 33]
    split_spikes_segs_50 = [[2, 4, 5, 10, 13, 18], np.array([22, 31, 33]) - len(spike_indices[0]["0"])]
    indices_in_51_seg_2 = np.array([20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34])
    split_spikes_segs_51 = [
        [0, 1, 3, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19],
        indices_in_51_seg_2 - len(spike_indices[0]["0"]),
    ]
    split_spikes_segs = [split_spikes_segs_50, split_spikes_segs_51]

    discard_spikes_curation_with_split = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "discard_spikes": [
            {
                "unit_id": "0",
                "indices": discard_spikes,
            }
        ],
        "splits": [{"unit_id": "0", "mode": "indices", "indices": [split_indices], "new_unit_ids": ["50", "51"]}],
    }

    # 4 and 13 should be discarded
    should_be_discarded_spikes = [
        [[4, 13], [22 - len(spike_indices[0]["0"])]],
        [[11], [30 - len(spike_indices[0]["0"])]],
    ]

    curated_sort_with_split = apply_curation(original_sort, discard_spikes_curation_with_split)

    for new_unit_id, discarded_spikes_each_unit, split_spikes_seg in zip(
        ["50", "51"], should_be_discarded_spikes, split_spikes_segs
    ):
        for segment_index in [0, 1]:

            original_spike_train = original_sort.get_unit_spike_train(unit_id="0", segment_index=segment_index)
            curated_spike_train = curated_sort_with_split.get_unit_spike_train(
                unit_id=new_unit_id, segment_index=segment_index
            )

            discard_spike_times = set(original_spike_train[split_spikes_seg[segment_index]]).difference(
                set(curated_spike_train)
            )

            spike_times_of_discarded_spikes = original_spike_train[discarded_spikes_each_unit[segment_index]]

            assert set(discard_spike_times) == set(spike_times_of_discarded_spikes)

    # test with "append" unit_id strategy

    split_indices = [2, 4, 5, 10, 13, 18, 22, 31, 33]
    split_spikes_segs_50 = [[2, 4, 5, 10, 13, 18], np.array([22, 31, 33]) - len(spike_indices[0]["0"])]
    indices_in_51_seg_2 = np.array([20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34])
    split_spikes_segs_51 = [
        [0, 1, 3, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19],
        indices_in_51_seg_2 - len(spike_indices[0]["0"]),
    ]
    split_spikes_segs = [split_spikes_segs_50, split_spikes_segs_51]

    discard_spikes_curation_with_split_append = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "discard_spikes": [
            {
                "unit_id": "0",
                "indices": discard_spikes,
            }
        ],
        "splits": [{"unit_id": "0", "mode": "indices", "indices": [split_indices]}],
    }

    # since we're using append, the new units will have id 6 and 7
    curated_sort_with_split_append = apply_curation(original_sort, discard_spikes_curation_with_split_append)

    for new_unit_id, discarded_spikes_each_unit, split_spikes_seg in zip(
        ["6", "7"], should_be_discarded_spikes, split_spikes_segs
    ):
        for segment_index in [0, 1]:

            original_spike_train = original_sort.get_unit_spike_train(unit_id="0", segment_index=segment_index)
            curated_spike_train = curated_sort_with_split_append.get_unit_spike_train(
                unit_id=new_unit_id, segment_index=segment_index
            )

            discard_spike_times = set(original_spike_train[split_spikes_seg[segment_index]]).difference(
                set(curated_spike_train)
            )

            spike_times_of_discarded_spikes = original_spike_train[discarded_spikes_each_unit[segment_index]]

            assert set(discard_spike_times) == set(spike_times_of_discarded_spikes)

    # since we're using append, the new units will have id 6 and 7
    curated_sort_with_split_split = apply_curation(
        original_sort, discard_spikes_curation_with_split_append, new_id_strategy="split"
    )

    for new_unit_id, discarded_spikes_each_unit, split_spikes_seg in zip(
        ["0-0", "0-1"], should_be_discarded_spikes, split_spikes_segs
    ):
        for segment_index in [0, 1]:

            original_spike_train = original_sort.get_unit_spike_train(unit_id="0", segment_index=segment_index)
            curated_spike_train = curated_sort_with_split_split.get_unit_spike_train(
                unit_id=new_unit_id, segment_index=segment_index
            )

            discard_spike_times = set(original_spike_train[split_spikes_seg[segment_index]]).difference(
                set(curated_spike_train)
            )

            spike_times_of_discarded_spikes = original_spike_train[discarded_spikes_each_unit[segment_index]]

            assert set(discard_spike_times) == set(spike_times_of_discarded_spikes)


def test_discard_and_split_several_units():

    spike_indices = [{unit_id: np.arange(10) for unit_id in [0, 1, 3, 6]}]
    original_sort = NumpySorting.from_unit_dict(spike_indices, sampling_frequency=1000)  # to have 1 sample=1ms

    discard_spikes = {0: [2, 4, 6, 8], 3: [1, 3, 5, 7, 9]}
    remaining_spikes = {
        0: [0, 1, 3, 5, 7, 9],
        3: [0, 2, 4, 6, 8],
    }

    split_spikes = {1: [5, 6, 8], 3: [2, 3, 4, 7, 8, 9]}

    expected_new_spikes = {
        0: [0, 1, 3, 5, 7, 9],
        6: np.arange(10),
        # 1 should split into 7 and 8
        7: [5, 6, 8],
        8: [0, 1, 2, 3, 4, 7, 9],
        # 3 should split into 9 and 10, without discarded spikes
        9: [2, 4, 8],
        10: [0, 6],
    }

    discard_spikes_curation = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "splits": [
            {"unit_id": 1, "mode": "indices", "indices": [split_spikes[1]]},
            {"unit_id": 3, "mode": "indices", "indices": [split_spikes[3]]},
        ],
        "discard_spikes": [
            {
                "unit_id": 0,
                "indices": discard_spikes[0],
            },
            {
                "unit_id": 3,
                "indices": discard_spikes[3],
            },
        ],
    }

    curated_sort = apply_curation(original_sort, discard_spikes_curation)

    for expected_unit_id, spikes in expected_new_spikes.items():
        assert np.all(curated_sort.get_unit_spike_train(unit_id=expected_unit_id) == spikes)


def test_discard_and_split_several_units_string_ids():

    spike_indices = [{unit_id: np.arange(10) for unit_id in ["0", "1", "3", "6"]}]
    original_sort = NumpySorting.from_unit_dict(spike_indices, sampling_frequency=1000)  # to have 1 sample=1ms

    discard_spikes = {0: [2, 4, 6, 8], 3: [1, 3, 5, 7, 9]}
    remaining_spikes = {
        0: [0, 1, 3, 5, 7, 9],
        3: [0, 2, 4, 6, 8],
    }

    split_spikes = {1: [5, 6, 8], 3: [2, 3, 4, 7, 8, 9]}

    expected_new_spikes = {
        "0": [0, 1, 3, 5, 7, 9],
        "6": np.arange(10),
        # 1 should split into 7 and 8
        "7": [5, 6, 8],
        "8": [0, 1, 2, 3, 4, 7, 9],
        # 3 should split into 9 and 10, without discarded spikes
        "9": [2, 4, 8],
        "10": [0, 6],
    }

    discard_spikes_curation = {
        "format_version": "3",
        "unit_ids": original_sort.unit_ids,
        "splits": [
            {"unit_id": "1", "mode": "indices", "indices": [split_spikes[1]]},
            {"unit_id": "3", "mode": "indices", "indices": [split_spikes[3]]},
        ],
        "discard_spikes": [
            {
                "unit_id": "0",
                "indices": discard_spikes[0],
            },
            {
                "unit_id": "3",
                "indices": discard_spikes[3],
            },
        ],
    }

    curated_sort = apply_curation(original_sort, discard_spikes_curation)

    for expected_unit_id, spikes in expected_new_spikes.items():
        assert np.all(curated_sort.get_unit_spike_train(unit_id=expected_unit_id) == spikes)

    expected_new_spikes_split = {
        "0": [0, 1, 3, 5, 7, 9],
        "6": np.arange(10),
        # 1 should split into "1-0" and "1-1"
        "1-0": [5, 6, 8],
        "1-1": [0, 1, 2, 3, 4, 7, 9],
        # 3 should split into "3-0" and "3-1", without discarded spikes
        "3-0": [2, 4, 8],
        "3-1": [0, 6],
    }

    curated_sort_split = apply_curation(original_sort, discard_spikes_curation, new_id_strategy="split")

    for expected_split_unit_id, spikes in expected_new_spikes_split.items():
        assert np.all(curated_sort_split.get_unit_spike_train(unit_id=expected_split_unit_id) == spikes)
