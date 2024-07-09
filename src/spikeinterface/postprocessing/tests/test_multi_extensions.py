import pytest

import time
import numpy as np

from spikeinterface import create_sorting_analyzer, generate_ground_truth_recording


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=7,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting


@pytest.fixture(scope="module")
def dataset():
    return get_dataset()


def test_SortingAnalyzer_merge_all_extensions(dataset):
    recording, sorting = dataset

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=False, sparsity=None)

    sorting_analyzer.compute(
        [
            "random_spikes",
            "noise_levels",
            "templates",
            "waveforms",
            "principal_components",
            "correlograms",
            "isi_histograms",
            "template_similarity",
            "spike_amplitudes",
            "amplitude_scalings",
            "spike_locations",
            "unit_locations",
            "template_metrics",
        ],
        n_jobs=1
    )  
    # return 
    # TODO fix n_jobs=1 should not trigger a warning but we have one!!

    merges = [[1, 2], [3, 4, 5]]

    t0 = time.perf_counter()
    analyzer_merged_h = sorting_analyzer.merge_units(merge_unit_groups=merges, censor_ms=5,
                                                     merging_mode="hard", n_jobs=1)
    t_h = time.perf_counter() - t0

    t0 = time.perf_counter()
    analyzer_merged_s = sorting_analyzer.merge_units(merge_unit_groups=merges, censor_ms=5, merging_mode="soft", 
                                                     sparsity_overlap=0.0, n_jobs=1)
    t_s = time.perf_counter() - t0
    # print(analyzer_merged_h)
    # print(analyzer_merged_s)

    assert t_s < t_h
    np.testing.assert_array_equal(analyzer_merged_h.unit_ids, analyzer_merged_s.unit_ids)
    np.testing.assert_array_equal(analyzer_merged_h.unit_ids, [0, 6, 7, 8])
    
    # TODO: add more tests
    


if __name__ == "__main__":
    dataset = get_dataset()
    test_SortingAnalyzer_merge_all_extensions(dataset)