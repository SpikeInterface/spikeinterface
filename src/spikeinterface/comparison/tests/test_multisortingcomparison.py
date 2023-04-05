import shutil
import pytest
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import NumpySorting, toy_example
from spikeinterface.comparison import compare_multiple_sorters, MultiSortingComparison

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "comparison"
else:
    cache_folder = Path("cache_folder") / "comparison"


multicomparison_folder = cache_folder / 'saved_multisorting_comparison'


def setup_module():
    if multicomparison_folder.is_dir():
        shutil.rmtree(multicomparison_folder)


def make_sorting(times1, labels1, times2, labels2, times3, labels3):
    sampling_frequency = 30000.
    sorting1 = NumpySorting.from_times_labels([times1], [labels1], sampling_frequency)
    sorting2 = NumpySorting.from_times_labels([times2], [labels2], sampling_frequency)
    sorting3 = NumpySorting.from_times_labels([times3], [labels3], sampling_frequency)
    sorting1 = sorting1.save()
    sorting2 = sorting2.save()
    sorting3 = sorting3.save()
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
    
    msc.save_to_folder(multicomparison_folder)
    
    msc = MultiSortingComparison.load_from_folder(multicomparison_folder)
    
    # import spikeinterface.widgets  as sw
    # import matplotlib.pyplot as plt
    # sw.plot_multicomp_graph(msc)
    # sw.plot_multicomp_agreement(msc)
    # sw.plot_multicomp_agreement_by_sorter(msc)
    # plt.show()


def test_compare_multi_segment():
    num_segments = 3
    _, sort = toy_example(num_segments=num_segments)

    cmp_multi = compare_multiple_sorters([sort, sort, sort])

    for k, cmp in cmp_multi.comparisons.items():
        assert np.allclose(np.diag(cmp.agreement_scores), np.ones(len(sort.unit_ids)))

    sort_agr = cmp_multi.get_agreement_sorting(minimum_agreement_count=num_segments)
    assert len(sort_agr.unit_ids) == len(sort.unit_ids)
    assert sort_agr.get_num_segments() == num_segments

    # test that all spike trains from different segments can be retrieved
    for unit in sort_agr.unit_ids:
        for seg_index in range(num_segments):
            st = sort_agr.get_unit_spike_train(unit, seg_index)
            print(f"Segment {seg_index} unit {unit}: {st}")


if __name__ == '__main__':
    test_compare_multiple_sorters()
