import os
import shutil

import pytest
import numpy as np
import time

from spikeinterface.core import generate_sorting
from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_multiple_sorters, MultiSortingComparison


ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


@pytest.fixture(scope="module")
def setup_module(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    multicomparison_folder = cache_folder / "saved_multisorting_comparison"
    if multicomparison_folder.is_dir():
        shutil.rmtree(multicomparison_folder)
    return multicomparison_folder


def make_sorting(times1, labels1, times2, labels2, times3, labels3):
    sampling_frequency = 30000.0
    sorting1 = NumpySorting.from_samples_and_labels([times1], [labels1], sampling_frequency)
    sorting2 = NumpySorting.from_samples_and_labels([times2], [labels2], sampling_frequency)
    sorting3 = NumpySorting.from_samples_and_labels([times3], [labels3], sampling_frequency)
    sorting1 = sorting1.save()
    sorting2 = sorting2.save()
    sorting3 = sorting3.save()
    return sorting1, sorting2, sorting3


def test_compare_multiple_sorters(setup_module):
    multicomparison_folder = setup_module
    # simple match
    sorting1, sorting2, sorting3 = make_sorting(
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [101, 201, 301, 400, 501, 598, 702, 801, 899, 1000, 1100, 2000, 3000],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4],
        [101, 201, 301, 400, 500, 600, 700, 800, 900, 1000, 1100, 2000, 3000, 3100, 3200, 3300],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5],
    )
    msc = compare_multiple_sorters([sorting1, sorting2, sorting3], verbose=True)
    msc_shuffle = compare_multiple_sorters([sorting3, sorting1, sorting2])
    msc_dist = compare_multiple_sorters([sorting3, sorting1, sorting2], agreement_method="distance")

    agr = msc._do_agreement_matrix()
    agr_shuffle = msc_shuffle._do_agreement_matrix()
    agr_dist = msc_dist._do_agreement_matrix()

    print(agr)
    print(agr_shuffle)
    print(agr_dist)

    assert len(msc.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids()) == 3
    assert len(msc.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids()) == 5
    assert len(msc.get_agreement_sorting().get_unit_ids()) == 6
    assert len(msc.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids()) == len(
        msc_shuffle.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids()
    )
    assert len(msc.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids()) == len(
        msc_shuffle.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids()
    )
    assert len(msc.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids()) == len(
        msc_dist.get_agreement_sorting(minimum_agreement_count=3).get_unit_ids()
    )
    assert len(msc.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids()) == len(
        msc_dist.get_agreement_sorting(minimum_agreement_count=2).get_unit_ids()
    )
    assert len(msc.get_agreement_sorting().get_unit_ids()) == len(msc_shuffle.get_agreement_sorting().get_unit_ids())
    assert len(msc.get_agreement_sorting().get_unit_ids()) == len(msc_dist.get_agreement_sorting().get_unit_ids())
    agreement_2 = msc.get_agreement_sorting(minimum_agreement_count=2, minimum_agreement_count_only=True)
    assert np.all([agreement_2.get_unit_property(u, "agreement_number")] == 2 for u in agreement_2.get_unit_ids())

    msc.save_to_folder(multicomparison_folder)

    msc = MultiSortingComparison.load_from_folder(multicomparison_folder)


def test_compare_multi_segment():
    num_segments = 3
    sort = generate_sorting(durations=[10] * num_segments)

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


def test_parallel():
    sorting = generate_sorting(durations=[3000])

    t_start = time.perf_counter()
    msc_1_job = compare_multiple_sorters([sorting] * 20, n_jobs=1)
    t_stop = time.perf_counter()
    elapsed_1_job = t_stop - t_start
    print(f"Elapsed 1 job: {elapsed_1_job}")

    t_start = time.perf_counter()
    msc_N_jobs = compare_multiple_sorters([sorting] * 20, n_jobs=-1)
    t_stop = time.perf_counter()
    elapsed_N_jobs = t_stop - t_start
    print(f"Elapsed N jobs: {elapsed_N_jobs}")

    # there is no guarantee there are more than 1 CPU on GH actions. Let's comment it out
    if not ON_GITHUB and os.cpu_count() > 1:
        assert elapsed_N_jobs < elapsed_1_job
    # check if the results are the same
    for k, cmp in msc_1_job.comparisons.items():
        cmp_N_jobs = msc_N_jobs.comparisons[k]
        np.testing.assert_array_equal(cmp.agreement_scores, cmp_N_jobs.agreement_scores)


if __name__ == "__main__":
    test_compare_multiple_sorters()
