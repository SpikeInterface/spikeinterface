import pytest

import spikeinterface.full as si
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import shutil

from spikeinterface.sortingcomponents.benchmark.tests.common_benchmark_testing import make_dataset, cache_folder
from spikeinterface.sortingcomponents.benchmark.benchmark_clustering import ClusteringStudy


@pytest.mark.skip()
def test_benchmark_clustering():

    job_kwargs = dict(n_jobs=0.8, chunk_duration="1s")

    recording, gt_sorting = make_dataset()

    num_spikes = gt_sorting.to_spike_vector().size
    spike_indices = np.arange(0, num_spikes, 5)

    # create study
    study_folder = cache_folder / "study_clustering"
    datasets = {"toy": (recording, gt_sorting)}
    cases = {}
    for method in ["random_projections", "circus"]:
        cases[method] = {
            "label": f"{method} on toy",
            "dataset": "toy",
            "init_kwargs": {"indices": spike_indices},
            "params": {"method": method, "method_kwargs": {}},
        }

    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = ClusteringStudy.create(study_folder, datasets=datasets, cases=cases)
    print(study)

    # this study needs analyzer
    study.create_sorting_analyzer_gt(**job_kwargs)
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
    # @pierre : This one has a bug
    # study.plot_metrics_vs_snr('cosine')
    study.homogeneity_score(ignore_noise=False)
    plt.show()


if __name__ == "__main__":
    test_benchmark_clustering()
