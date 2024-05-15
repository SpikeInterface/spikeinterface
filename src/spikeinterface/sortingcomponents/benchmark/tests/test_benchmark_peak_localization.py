import pytest

import shutil

import matplotlib.pyplot as plt


from spikeinterface.sortingcomponents.benchmark.tests.common_benchmark_testing import make_dataset, cache_folder

from spikeinterface.sortingcomponents.benchmark.benchmark_peak_localization import PeakLocalizationStudy
from spikeinterface.sortingcomponents.benchmark.benchmark_peak_localization import UnitLocalizationStudy


@pytest.mark.skip()
def test_benchmark_peak_localization():
    job_kwargs = dict(n_jobs=0.8, chunk_duration="100ms")

    # recording, gt_sorting = make_dataset()
    recording, gt_sorting, gt_analyzer = make_dataset()

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
                "method": method,
                "method_kwargs": {"ms_before": 2},
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
    study.plot_comparison_positions()
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
            "params": {"method": method, "method_kwargs": {"ms_before": 2}},
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
    study.plot_comparison_positions()
    study.plot_template_errors()
    study.plot_run_times()

    plt.show()


if __name__ == "__main__":
    # test_benchmark_peak_localization()
    test_benchmark_unit_localization()
