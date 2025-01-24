import pytest
from pathlib import Path
import numpy as np

import shutil

from spikeinterface.benchmark.benchmark_merging import MergingStudy
from spikeinterface.benchmark.tests.common_benchmark_testing import make_dataset

from spikeinterface.generation.splitting_tools import split_sorting_by_amplitudes, split_sorting_by_times


@pytest.mark.skip()
def test_benchmark_merging(create_cache_folder):
    cache_folder = create_cache_folder
    job_kwargs = dict(n_jobs=0.8, chunk_duration="1s")

    recording, gt_sorting, gt_analyzer = make_dataset(job_kwargs)

    # create study
    study_folder = cache_folder / "study_clustering"
    # datasets = {"toy": (recording, gt_sorting)}
    datasets = {"toy": gt_analyzer}

    gt_analyzer.compute(["random_spikes", "templates", "spike_amplitudes"])

    splitted_sorting = {}
    splitted_sorting["times"] = split_sorting_by_times(gt_analyzer)
    splitted_sorting["amplitudes"] = split_sorting_by_amplitudes(gt_analyzer)

    cases = {}
    for splits in ["times", "amplitudes"]:
        for method in ["circus", "lussac"]:
            cases[(method, splits)] = {
                "label": f"{method}",
                "dataset": "toy",
                "init_kwargs": {"gt_sorting": gt_sorting, "splitted_cells": splitted_sorting[splits][1]},
                "params": {"method": method, "splitted_sorting": splitted_sorting[splits][0], "method_kwargs": {}},
            }

    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = MergingStudy.create(study_folder, datasets=datasets, cases=cases)
    study.compute_metrics()
    study.run(**job_kwargs)
    study.compute_results()
    # plots
    # study.plot_performances_vs_snr()
    study.plot_agreements()
    study.plot_unit_counts()
    # study.plot_error_metrics()
    # study.plot_metrics_vs_snr()
    # study.plot_run_times()
    # study.plot_metrics_vs_snr("cosine")
    # study.homogeneity_score(ignore_noise=False)
    # import matplotlib.pyplot as plt
    # plt.show()


if __name__ == "__main__":
    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks"
    test_benchmark_merging(cache_folder)
