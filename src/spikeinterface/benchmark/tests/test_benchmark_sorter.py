import shutil
import pytest
from pathlib import Path

from spikeinterface import generate_ground_truth_recording
from spikeinterface.preprocessing import bandpass_filter
from spikeinterface.benchmark import SorterStudy


@pytest.fixture(scope="module")
def create_simple_study(tmp_path_factory):
    study_folder = tmp_path_factory.mktemp("sorter_study_folder")
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    _create_simple_study(study_folder)
    return study_folder


@pytest.fixture(scope="module")
def create_complex_study(tmp_path_factory):
    study_folder = tmp_path_factory.mktemp("sorter_study_folder_complex")
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    _create_complex_study(study_folder)
    return study_folder


def simple_preprocess(rec):
    return bandpass_filter(rec)


def _create_simple_study(study_folder):
    rec0, gt_sorting0 = generate_ground_truth_recording(num_channels=4, durations=[30.0], seed=42)
    rec1, gt_sorting1 = generate_ground_truth_recording(num_channels=4, durations=[30.0], seed=91)

    datasets = {
        "toy_tetrode": (rec0, gt_sorting0),
        "toy_probe32": (rec1, gt_sorting1),
        "toy_probe32_preprocess": (simple_preprocess(rec1), gt_sorting1),
    }

    # cases can also be generated via simple loops
    cases = {
        #
        ("tdc2", "no-preprocess", "tetrode"): {
            "label": "tridesclous2 without preprocessing and standard params",
            "dataset": "toy_tetrode",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        #
        ("tdc2", "with-preprocess", "probe32"): {
            "label": "tridesclous2 with preprocessing standar params",
            "dataset": "toy_probe32_preprocess",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
    }

    study = SorterStudy.create(
        study_folder, datasets=datasets, cases=cases, levels=["sorter_name", "processing", "probe_type"]
    )
    # print(study)


def _create_complex_study(study_folder):
    rec0, gt_sorting0 = generate_ground_truth_recording(num_channels=4, durations=[30.0], seed=42)
    rec1, gt_sorting1 = generate_ground_truth_recording(num_channels=4, durations=[30.0], seed=91)
    rec2, gt_sorting2 = generate_ground_truth_recording(num_channels=4, durations=[30.0], seed=91)
    rec3, gt_sorting3 = generate_ground_truth_recording(num_channels=4, durations=[30.0], seed=91)

    datasets = {
        "toy_tetrode": (rec0, gt_sorting0),
        "toy_tetrode_preprocess": (simple_preprocess(rec0), gt_sorting0),
        "toy_probe32": (rec1, gt_sorting1),
        "toy_probe32_preprocess": (simple_preprocess(rec1), gt_sorting1),
        "toy_probe64": (rec2, gt_sorting2),
        "toy_probe64_preprocess": (simple_preprocess(rec2), gt_sorting2),
        "toy_probe256": (rec3, gt_sorting3),
        "toy_probe256_preprocess": (simple_preprocess(rec3), gt_sorting3),
    }

    # cases can also be generated via simple loops
    cases = {
        ("tdc2", "no-preprocess", "tetrode"): {
            "label": "tridesclous2 without preprocessing and standard params",
            "dataset": "toy_tetrode",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("tdc2", "preprocess", "tetrode"): {
            "label": "tridesclous2 with preprocessing and standard params",
            "dataset": "toy_tetrode_preprocess",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("tdc2", "no-preprocess", "probe32"): {
            "label": "tridesclous2 without preprocessing and standard params",
            "dataset": "toy_probe32",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("tdc2", "preprocess", "probe32"): {
            "label": "tridesclous2 with preprocessing and standard params",
            "dataset": "toy_probe32_preprocess",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("tdc2", "no-preprocess", "probe64"): {
            "label": "tridesclous2 without preprocessing and standard params",
            "dataset": "toy_probe64",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("tdc2", "preprocess", "probe64"): {
            "label": "tridesclous2 with preprocessing and standard params",
            "dataset": "toy_probe64_preprocess",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("tdc2", "no-preprocess", "probe256"): {
            "label": "tridesclous2 without preprocessing and standard params",
            "dataset": "toy_probe256",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("tdc2", "preprocess", "probe256"): {
            "label": "tridesclous2 with preprocessing and standard params",
            "dataset": "toy_probe256_preprocess",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        ("sc2", "no-preprocess", "tetrode"): {
            "label": "spykingcircus2 without preprocessing and standard params",
            "dataset": "toy_tetrode",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
        ("sc2", "preprocess", "tetrode"): {
            "label": "spykingcircus2 with preprocessing and standard params",
            "dataset": "toy_tetrode_preprocess",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
        ("sc2", "no-preprocess", "probe32"): {
            "label": "spykingcircus2 without preprocessing and standard params",
            "dataset": "toy_probe32",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
        ("sc2", "preprocess", "probe32"): {
            "label": "spykingcircus2 with preprocessing and standard params",
            "dataset": "toy_probe32_preprocess",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
        ("sc2", "no-preprocess", "probe64"): {
            "label": "spykingcircus2 without preprocessing and standard params",
            "dataset": "toy_probe64",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
        ("sc2", "preprocess", "probe64"): {
            "label": "spykingcircus2 with preprocessing and standard params",
            "dataset": "toy_probe64_preprocess",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
        ("sc2", "no-preprocess", "probe256"): {
            "label": "spykingcircus2 without preprocessing and standard params",
            "dataset": "toy_probe256",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
        ("sc2", "preprocess", "probe256"): {
            "label": "spykingcircus2 with preprocessing and standard params",
            "dataset": "toy_probe256_preprocess",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
    }

    study = SorterStudy.create(
        study_folder, datasets=datasets, cases=cases, levels=["sorter_name", "processing", "probe_type"]
    )


def test_SorterStudy(create_simple_study):
    # job_kwargs = dict(n_jobs=2, chunk_duration="1s")

    study_folder = create_simple_study
    study = SorterStudy(study_folder)
    print(study)

    # # this run the sorters
    study.run()

    # # this run comparisons
    study.compute_results()
    print(study)

    # this is from the base class
    rt = study.get_run_times()
    # rt = study.plot_run_times()
    # import matplotlib.pyplot as plt
    # plt.show()

    perf_by_unit = study.get_performance_by_unit()
    # print(perf_by_unit)
    count_units = study.get_count_units()
    # print(count_units)


def test_get_grouped_keys_mapping(create_complex_study):
    study_folder = create_complex_study
    study = SorterStudy(study_folder)

    keys, _ = study.get_grouped_keys_mapping()
    assert len(keys) == len(study.cases)

    keys, _ = study.get_grouped_keys_mapping(levels_to_group_by=["sorter_name"])
    assert len(keys) == 2

    keys, _ = study.get_grouped_keys_mapping(levels_to_group_by=["processing"])
    assert len(keys) == 2

    keys, _ = study.get_grouped_keys_mapping(levels_to_group_by=["probe_type"])
    assert len(keys) == 4

    keys, _ = study.get_grouped_keys_mapping(levels_to_group_by=["sorter_name", "processing"])
    assert len(keys) == 4

    keys, _ = study.get_grouped_keys_mapping(levels_to_group_by=["sorter_name", "probe_type"])
    assert len(keys) == 8

    keys, _ = study.get_grouped_keys_mapping(levels_to_group_by=["processing", "probe_type"])
    assert len(keys) == 8

    keys, _ = study.get_grouped_keys_mapping(levels_to_group_by=["sorter_name", "processing", "probe_type"])
    assert len(keys) == 16


if __name__ == "__main__":
    study_folder_simple = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks" / "test_SorterStudy"
    if study_folder_simple.exists():
        shutil.rmtree(study_folder_simple)
    _create_simple_study(study_folder_simple)
    test_SorterStudy(study_folder_simple)
    study_folder_complex = (
        Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks" / "test_SorterStudy_complex"
    )
    if study_folder_complex.exists():
        shutil.rmtree(study_folder_complex)
    _create_complex_study(study_folder_complex)
    test_get_grouped_keys_mapping(study_folder_complex)

# # test out all plots and levels
# import matplotlib.pyplot as plt

# from spikeinterface.benchmark.benchmark_plot_tools import (
#     plot_run_times,
#     plot_performances_ordered,
#     plot_performances_swarm,
#     plot_performances_vs_snr,
#     plot_performances_vs_depth_and_snr,
#     plot_performances_comparison,
#     plot_unit_counts,
# )

# study_folder_complex = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks" / "test_SorterStudy_complex"
# if not study_folder_complex.is_dir():
#     _create_complex_study(study_folder_complex)
# study = SorterStudy(study_folder_complex)

# study.compute_metrics()

# plot_funcs = [
#     plot_run_times,
#     plot_performances_ordered,
#     plot_performances_swarm,
#     plot_performances_vs_snr,
#     plot_performances_vs_depth_and_snr,
#     plot_performances_comparison,
#     plot_unit_counts
# ]
# levels = [["sorter_name"], ["sorter_name", "processing"], ["sorter_name", "processing", "probe_type"]]

# for plot_func in plot_funcs:
#     for level in levels:
#         fig = plot_func(study, levels_to_keep=level)
#         fig.suptitle(f"{plot_func.__name__} - {level}")

# plt.ion()
# plt.show()
