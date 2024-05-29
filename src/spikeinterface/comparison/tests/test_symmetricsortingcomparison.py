import numpy as np

from spikeinterface.core import generate_sorting
from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_two_sorters


def make_sorting(times1, labels1, times2, labels2):
    sampling_frequency = 30000.0
    sorting1 = NumpySorting.from_times_labels([times1], [labels1], sampling_frequency)
    sorting2 = NumpySorting.from_times_labels([times2], [labels2], sampling_frequency)
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
    sc = compare_two_sorters(sorting1, sorting2)

    print(sc.agreement_scores)


def test_compare_multi_segment():
    sort = generate_sorting(durations=[10, 10])

    cmp_multi = compare_two_sorters(sort, sort)

    assert np.allclose(np.diag(cmp_multi.agreement_scores), np.ones(len(sort.unit_ids)))


if __name__ == "__main__":
    test_compare_two_sorters()
    test_compare_multi_segment()
