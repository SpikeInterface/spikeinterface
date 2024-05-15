import pytest

import shutil

import spikeinterface.full as si
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from spikeinterface.sortingcomponents.benchmark.tests.common_benchmark_testing import make_dataset, cache_folder
from spikeinterface.sortingcomponents.benchmark.benchmark_peak_detection import PeakDetectionStudy
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.template_tools import get_template_extremum_channel


@pytest.mark.skip()
def test_benchmark_peak_detection():
    job_kwargs = dict(n_jobs=0.8, chunk_duration="100ms")

    # recording, gt_sorting = make_dataset()
    recording, gt_sorting, gt_analyzer = make_dataset()

    # create study
    study_folder = cache_folder / "study_peak_detection"
    datasets = {"toy": (recording, gt_sorting)}
    cases = {}

    peaks = {}
    for dataset in datasets.keys():

        recording, gt_sorting = datasets[dataset]

        sorting_analyzer = create_sorting_analyzer(gt_sorting, recording, format="memory", sparse=False)
        sorting_analyzer.compute(["random_spikes", "templates"])
        extremum_channel_inds = get_template_extremum_channel(sorting_analyzer, outputs="index")
        spikes = gt_sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        peaks[dataset] = spikes

    for method in ["locally_exclusive", "by_channel"]:
        cases[method] = {
            "label": f"{method} on toy",
            "dataset": "toy",
            "init_kwargs": {"gt_peaks": peaks["toy"]},
            "params": {"ms_before": 2, "method": method, "method_kwargs": {}},
        }

    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = PeakDetectionStudy.create(study_folder, datasets=datasets, cases=cases)
    print(study)

    # this study needs analyzer
    study.create_sorting_analyzer_gt(**job_kwargs)
    study.compute_metrics()

    # run and result
    study.run(**job_kwargs)
    study.compute_results()

    # load study to check persistency
    study = PeakDetectionStudy(study_folder)
    study.plot_agreements_by_channels()
    study.plot_agreements_by_units()
    study.plot_deltas_per_cells()
    study.plot_detected_amplitudes()
    study.plot_performances_vs_snr()
    study.plot_template_similarities()
    study.plot_run_times()

    plt.show()


if __name__ == "__main__":
    # test_benchmark_peak_localization()
    test_benchmark_peak_detection()
