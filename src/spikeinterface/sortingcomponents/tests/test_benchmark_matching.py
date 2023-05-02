import pytest
import numpy as np
import pandas as pd
import shutil
import os
from pathlib import Path

import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface.sortingcomponents.benchmark import benchmark_matching

@pytest.fixture(scope="session")
def benchmark_and_kwargs():
    recording, sorting = se.toy_example(duration=1, num_channels=2, num_units=2, num_segments=1,
                                        firing_rate=10, seed=0)
    recording = spre.common_reference(recording, dtype='float32')
    cwd = Path.cwd()
    we_path = cwd / "waveforms"
    sort_path = cwd / "sorting.npz"
    se.NpzSortingExtractor.write_sorting(sorting, sort_path)
    sorting = se.NpzSortingExtractor(sort_path)
    we = sc.extract_waveforms(recording, sorting, we_path, overwrite=True)
    templates = we.get_all_templates()
    noise_levels = sc.get_noise_levels(recording, return_scaled=False)
    methods_kwargs = {
        'tridesclous': dict(waveform_extractor=we, noise_levels=noise_levels),
        'wobble': dict(templates=templates, nbefore=we.nbefore, nafter=we.nafter,
                       parameters={'approx_rank' : 2})
    }
    methods = list(methods_kwargs.keys())
    benchmark = benchmark_matching.BenchmarkMatching(recording, sorting, we, methods, methods_kwargs)
    yield benchmark, methods_kwargs

    # Teardown
    shutil.rmtree(we_path)
    os.remove(sort_path)

def test_run_matching_vary_parameter(benchmark_and_kwargs):
    # Arrange
    benchmark, methods_kwargs = benchmark_and_kwargs
    parameter_names = list(benchmark.parameter_name2matching_fn.keys())
    num_spikes = [1, 10, 100]
    fractions = [0, 0.5, 1]
    parameter_sets = [num_spikes, fractions, fractions]
    num_replicates = 2

    for parameters, parameter_name in zip(parameter_sets, parameter_names):
        # Act
        with benchmark as bmk:
            matching_df = bmk.run_matching_vary_parameter(parameters, parameter_name, num_replicates=num_replicates)

        # Assert
        assert matching_df.shape[0] == len(parameters) * num_replicates * len(methods_kwargs)
        assert matching_df.shape[1] == 6

    # Check invalid inputs
    with benchmark as bmk:
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([1, 2], 'invalid_parameter_name')
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([1, 2], 'num_spikes', num_replicates=-1)
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([1, 2], 'num_spikes', num_replicates=0.5)
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([0, -1], 'num_spikes')
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([-1, 2], 'fraction_misclassed')
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([0, 1], 'fraction_misclassed', min_similarity=2)
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([-1, 2], 'fraction_missing')
        with pytest.raises(ValueError):
            bmk.run_matching_vary_parameter([0, 1], 'fraction_missing', snr_threshold=-1)

def test_compare_all_sortings(benchmark_and_kwargs):
    # Arrange
    benchmark, methods_kwargs = benchmark_and_kwargs
    parameter_name = "num_spikes"
    num_replicates = 2
    num_spikes = [1, 10, 100]
    np.random.seed(0)
    sortings, gt_sortings, parameter_values, parameter_names, iter_nums, methods = [], [], [], [], [], []
    for replicate in range(num_replicates):
        for spike_num in num_spikes:
            for method in list(methods_kwargs.keys()):
                len_spike_train = 100
                spike_time_inds = np.random.choice(benchmark.recording.get_num_frames(), len_spike_train,
                                                   replace=False)
                unit_ids = np.random.choice(benchmark.gt_sorting.get_unit_ids(), len_spike_train,
                                            replace=True)
                sort_index = np.argsort(spike_time_inds)
                spike_time_inds = spike_time_inds[sort_index]
                unit_ids = unit_ids[sort_index]
                sorting = sc.NumpySorting.from_times_labels(spike_time_inds, unit_ids,
                                                            benchmark.recording.sampling_frequency)
                spike_time_inds = np.random.choice(benchmark.recording.get_num_frames(), len_spike_train,
                                                   replace=False)
                unit_ids = np.random.choice(benchmark.gt_sorting.get_unit_ids(), len_spike_train,
                                            replace=True)
                sort_index = np.argsort(spike_time_inds)
                spike_time_inds = spike_time_inds[sort_index]
                unit_ids = unit_ids[sort_index]
                gt_sorting = sc.NumpySorting.from_times_labels(spike_time_inds, unit_ids,
                                                               benchmark.recording.sampling_frequency)
                sortings.append(sorting)
                gt_sortings.append(gt_sorting)
                parameter_values.append(spike_num)
                parameter_names.append(parameter_name)
                iter_nums.append(replicate)
                methods.append(method)
    matching_df = pd.DataFrame({'sorting': sortings,
                                'gt_sorting': gt_sortings,
                                'parameter_value': parameter_values,
                                'parameter_name': parameter_names,
                                'iter_num': iter_nums,
                                'method': methods})
    comparison_from_df = matching_df.copy()
    comparison_from_self = matching_df.copy()
    comparison_collision = matching_df.copy()

    # Act
    benchmark.compare_all_sortings(comparison_from_df, ground_truth='from_df')
    benchmark.compare_all_sortings(comparison_from_self, ground_truth='from_self')
    benchmark.compare_all_sortings(comparison_collision, collision=True)

    # Assert
    for comparison in [comparison_from_df, comparison_from_self, comparison_collision]:
        assert comparison.shape[0] == len(num_spikes) * num_replicates * len(methods_kwargs)
        assert comparison.shape[1] == 7
        for comp, sorting in zip(comparison['comparison'], comparison['sorting']):
            comp.sorting2 == sorting
    for comp, gt_sorting in zip(comparison_from_df['comparison'], comparison['gt_sorting']):
        comp.sorting1 == gt_sorting
    for comp in comparison_from_self['comparison']:
        comp.sorting1 == benchmark.gt_sorting

    # Check invalid inputs
    with pytest.raises(ValueError):
        benchmark.compare_all_sortings(comparison_from_df, ground_truth='invalid')

if __name__ == '__main__':
    test_run_matching_vary_parameter(benchmark_and_kwargs)
    test_compare_all_sortings(benchmark_and_kwargs)
