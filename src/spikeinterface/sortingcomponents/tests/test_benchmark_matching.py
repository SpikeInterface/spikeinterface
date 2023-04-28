import pytest
import numpy as np
import shutil
import os
from pathlib import Path

import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface.sortingcomponents.benchmark import benchmark_matching

@pytest.fixture(scope="session")
def benchmarks_and_kwargs():
    recording, sorting = se.toy_example(seed=0, num_segments=1, num_channels=10)
    recording = spre.common_reference(recording, dtype='float32')
    cwd = Path.cwd()
    we_path = cwd / "waveforms"
    sort_path = cwd / "sorting.npz"
    se.NpzSortingExtractor.write_sorting(sorting, sort_path)
    sorting = se.NpzSortingExtractor(sort_path)
    we = sc.extract_waveforms(recording, sorting, we_path, overwrite=True)
    templates = we.get_all_templates()
    noise_levels = sc.get_noise_levels(recording, return_scaled=False)
    # TODO : Fix circus-omp
    methods_kwargs = {
        'naive': dict(waveform_extractor=we, noise_levels=noise_levels),
        'tridesclous': dict(waveform_extractor=we, noise_levels=noise_levels),
        'circus': dict(waveform_extractor=we, noise_levels=noise_levels),
        # 'circus-omp': dict(waveform_extractor=we, noise_levels=noise_levels),
        'wobble': dict(templates=templates, nbefore=we.nbefore, nafter=we.nafter)
    }
    methods = list(methods_kwargs.keys())
    benchmark = benchmark_matching.BenchmarkMatching(recording, sorting, we, methods, methods_kwargs)
    benchmark_not_exhaustive = benchmark_matching.BenchmarkMatching(recording, sorting, we, methods, methods_kwargs,
                                                                    exhaustive_gt=False)
    yield benchmark, benchmark_not_exhaustive, methods_kwargs

    # Teardown
    shutil.rmtree(we_path)
    os.remove(sort_path)

def test_run_matching(benchmarks_and_kwargs):
    benchmark, benchmark_not_exhaustive, methods_kwargs = benchmarks_and_kwargs
    benchmark.run_matching(methods_kwargs, collision=False)
    # for benchmark in [benchmark, benchmark_not_exhaustive]:
    #     benchmark.run_matching(methods_kwargs, collision=False)
    #     benchmark.run_matching(methods_kwargs, collision=True)

if __name__ == '__main__':
    test_run_matching(benchmarks_and_kwargs)
