import pytest
import numpy as np
from numpy.testing import assert_array_equal

from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_multiple_sorters


def make_sorting(times1, labels1, times2, labels2, times3, labels3):
    sampling_frequency = 30000.
    sorting1 = NumpySorting.from_times_labels([times1], [labels1], sampling_frequency)
    sorting2 = NumpySorting.from_times_labels([times2], [labels2], sampling_frequency)
    sorting3 = NumpySorting.from_times_labels([times3], [labels3], sampling_frequency)
    return sorting1, sorting2, sorting3


def test_compare_multiple_sorters():
    # simple match
    sorting1, sorting2, sorting3 = make_sorting([100, 200, 300, 400, 500, 600, 700, 800, 900],
                                                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                                                [101, 201, 301, 400, 501, 598, 702, 801, 899, 1000, 1100, 2000, 3000],
                                                [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4],
                                                [101, 201, 301, 400, 500, 600, 700, 800, 900, 1000, 1100, 2000, 3000,
                                                 3100, 3200, 3300],
                                                [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5], )
    msc = compare_multiple_sorters([sorting1, sorting2, sorting3], verbose=True)
    msc_shuffle = compare_multiple_sorters([sorting3, sorting1, sorting2])

    agr = msc._do_agreement_matrix()
    agr_shuffle = msc_shuffle._do_agreement_matrix()

    print(agr)
    print(agr_shuffle)

    assert len(msc.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids()) == 3
    assert len(msc.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids()) == 5
    assert len(msc.get_agreement_sorting().get_unit_ids()) == 6
    assert len(msc.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids()) == \
           len(msc_shuffle.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids())
    assert len(msc.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids()) == \
           len(msc_shuffle.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids())
    assert len(msc.get_agreement_sorting().get_unit_ids()) == len(msc_shuffle.get_agreement_sorting().get_unit_ids())
    agreement_2 = msc.get_agreement_sorting(minimum_agreement_count=2, minimum_agreement_count_only=True)
    assert np.all([agreement_2.get_unit_property(u, 'agreement_number')] == 2 for u in agreement_2.get_unit_ids())


if __name__ == '__main__':
    test_compare_multiple_sorters()
