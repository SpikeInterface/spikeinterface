import pytest

import shutil

import spikeinterface.full as si
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


from spikeinterface.sortingcomponents.benchmark.tests.common_benchmark_testing import make_dataset, cache_folder

from spikeinterface.sortingcomponents.benchmark.benchmark_peak_localization import PeakLocalizationStudy

from spikeinterface.sortingcomponents.benchmark.benchmark_peak_localization import UnitLocalizationStudy


@pytest.mark.skip()
def test_benchmark_peak_localization():
    job_kwargs = dict(n_jobs=0.8, chunk_duration="100ms")

    recording, gt_sorting = make_dataset()

    # create study
    study_folder = cache_folder / "study_peak_localization"
    datasets = {"toy": (recording, gt_sorting)}
    cases = {}
    for method in ["center_of_mass", "grid_convolution", "monopolar_triangulation"]:
        cases[method] = {
            "label": f"{method} on toy",
            "dataset": "toy",
            "init_kwargs": {"gt_positions": gt_sorting.get_property("gt_unit_locations")},
            "params": {
                "ms_before": 2,
                "method": method,
                "method_kwargs": {},
                "spike_retriver_kwargs": {"channel_from_template": False},
            },
        }

    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = PeakLocalizationStudy.create(study_folder, datasets=datasets, cases=cases)
    print(study)

    # this study needs analyzer
    study.create_sorting_analyzer_gt(**job_kwargs)
    study.compute_metrics()

    # run and result
    study.run(**job_kwargs)
    study.compute_results()

    # load study to check persistency
    study = PeakLocalizationStudy(study_folder)
    study.plot_comparison_positions(smoothing_factor=31)
    study.plot_run_times()

    plt.show()


@pytest.mark.skip()
def test_benchmark_unit_localization():
    job_kwargs = dict(n_jobs=0.8, chunk_duration="100ms")

    recording, gt_sorting = make_dataset()

    # create study
    study_folder = cache_folder / "study_unit_localization"
    datasets = {"toy": (recording, gt_sorting)}
    cases = {}
    for method in ["center_of_mass", "grid_convolution", "monopolar_triangulation"]:
        cases[method] = {
            "label": f"{method} on toy",
            "dataset": "toy",
            "init_kwargs": {"gt_positions": gt_sorting.get_property("gt_unit_locations")},
            "params": {
                "ms_before": 2,
                "method": method,
                "method_kwargs": {},
                "spike_retriver_kwargs": {"channel_from_template": False},
            },
        }

    if study_folder.exists():
        shutil.rmtree(study_folder)
    study = UnitLocalizationStudy.create(study_folder, datasets=datasets, cases=cases)
    print(study)

    # this study needs analyzer
    study.create_sorting_analyzer_gt(**job_kwargs)
    study.compute_metrics()

    # run and result
    study.run(**job_kwargs)
    study.compute_results()

    # load study to check persistency
    study = UnitLocalizationStudy(study_folder)
    study.plot_comparison_positions(smoothing_factor=31)
    study.plot_run_times()

    plt.show()


if __name__ == "__main__":
    # test_benchmark_peak_localization()
    test_benchmark_unit_localization()
