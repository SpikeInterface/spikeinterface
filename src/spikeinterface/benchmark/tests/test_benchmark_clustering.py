import pytest
import numpy as np

import shutil

from spikeinterface.benchmark.tests.common_benchmark_testing import make_dataset
from spikeinterface.benchmark.benchmark_clustering import ClusteringStudy
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.template_tools import get_template_extremum_channel

from pathlib import Path


@pytest.mark.skip()
def test_benchmark_clustering(create_cache_folder):
    cache_folder = create_cache_folder
    job_kwargs = dict(n_jobs=0.8, chunk_duration="1s")

    recording, gt_sorting, gt_analyzer = make_dataset()

    num_spikes = gt_sorting.to_spike_vector().size
    spike_indices = np.arange(0, num_spikes, 5)

    # create study
    study_folder = cache_folder / "study_clustering"
    # datasets = {"toy": (recording, gt_sorting)}
    datasets = {"toy": gt_analyzer}

    peaks = {}
    for dataset, gt_analyzer in datasets.items():

        # recording, gt_sorting = datasets[dataset]

        # sorting_analyzer = create_sorting_analyzer(gt_sorting, recording, format="memory", sparse=False)
        # sorting_analyzer.compute(["random_spikes", "templates"])
        extremum_channel_inds = get_template_extremum_channel(gt_analyzer, outputs="index")
        spikes = gt_analyzer.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        peaks[dataset] = spikes

    cases = {}
    for method in ["random_projections", "circus", "tdc_clustering"]:
        cases[method] = {
            "label": f"{method} on toy",
            "dataset": "toy",
            "init_kwargs": {"indices": spike_indices, "peaks": peaks["toy"]},
            "params": {"method": method, "method_kwargs": {}},
        }

    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = ClusteringStudy.create(study_folder, datasets=datasets, cases=cases)
    print(study)

    # this study needs analyzer
    # study.create_sorting_analyzer_gt(**job_kwargs)
    study.compute_metrics()

    study = ClusteringStudy(study_folder)

    # run and result
    study.run(**job_kwargs)
    study.compute_results()

    # load study to check persistency
    study = ClusteringStudy(study_folder)
    print(study)

    # plots
    study.plot_performances_vs_snr()
    study.plot_agreements()
    study.plot_comparison_clustering()
    study.plot_error_metrics()
    study.plot_metrics_vs_snr()
    study.plot_run_times()
    study.plot_metrics_vs_snr("cosine")
    study.homogeneity_score(ignore_noise=False)
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks"
    test_benchmark_clustering(cache_folder)
