import numpy as np

from spikeinterface.core import generate_sorting, aggregate_units
from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_two_sorters


def make_sorting(times1, labels1, times2, labels2):
    sampling_frequency = 30000.0
    sorting1 = NumpySorting.from_samples_and_labels([times1], [labels1], sampling_frequency)
    sorting2 = NumpySorting.from_samples_and_labels([times2], [labels2], sampling_frequency)
    return sorting1, sorting2


def test_compare_two_sorters():
    # simple match
    sorting1, sorting2 = make_sorting(
        [100, 200, 300, 400],
        [0, 0, 1, 0],
        [
            101,
            201,
            301,
        ],
        [0, 0, 5],
    )
    sc_from_counts = compare_two_sorters(sorting1, sorting2, agreement_method="count")
    sc_from_distance = compare_two_sorters(sorting1, sorting2, agreement_method="distance")

    np.testing.assert_array_equal(
        sc_from_counts.hungarian_match_12.to_numpy(),
        sc_from_distance.hungarian_match_12.to_numpy(),
    )


def test_compare_multi_segment():
    sort = generate_sorting(durations=[10, 10])

    cmp_multi = compare_two_sorters(sort, sort)

    assert np.allclose(np.diag(cmp_multi.agreement_scores), np.ones(len(sort.unit_ids)))


def test_agreements():
    """
    Test that the agreement scores are the same when using from_count and distance_matrix
    """
    sorting1 = generate_sorting(num_units=100)
    sorting_extra = generate_sorting(num_units=50)
    sorting2 = aggregate_units([sorting1, sorting_extra])
    sorting2 = sorting2.select_units(unit_ids=sorting2.unit_ids[np.random.permutation(len(sorting2.unit_ids))])

    sc_from_counts = compare_two_sorters(sorting1, sorting2, agreement_method="count")
    sc_from_distance = compare_two_sorters(sorting1, sorting2, agreement_method="distance")

    np.testing.assert_array_equal(
        sc_from_counts.hungarian_match_12.to_numpy(),
        sc_from_distance.hungarian_match_12.to_numpy(),
    )


if __name__ == "__main__":
    test_compare_two_sorters()
    test_compare_multi_segment()
