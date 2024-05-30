import shutil
import pytest
from pathlib import Path

from spikeinterface import generate_ground_truth_recording
from spikeinterface.preprocessing import bandpass_filter
from spikeinterface.comparison import GroundTruthStudy


@pytest.fixture(scope="module")
def setup_module(tmp_path_factory):
    study_folder = tmp_path_factory.mktemp("study_folder")
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    create_a_study(study_folder)
    return study_folder


def simple_preprocess(rec):
    return bandpass_filter(rec)


def create_a_study(study_folder):
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
            "run_sorter_params": {
                "sorter_name": "tridesclous2",
            },
            "comparison_params": {},
        },
        #
        ("tdc2", "with-preprocess", "probe32"): {
            "label": "tridesclous2 with preprocessing standar params",
            "dataset": "toy_probe32_preprocess",
            "run_sorter_params": {
                "sorter_name": "tridesclous2",
            },
            "comparison_params": {},
        },
        # we comment this at the moement because SC2 is quite slow for testing
        # ("sc2", "no-preprocess", "tetrode"): {
        #     "label": "spykingcircus2 without preprocessing standar params",
        #     "dataset": "toy_tetrode",
        #     "run_sorter_params": {
        #         "sorter_name": "spykingcircus2",
        #     },
        #     "comparison_params": {
        #     },
        # },
    }

    study = GroundTruthStudy.create(
        study_folder, datasets=datasets, cases=cases, levels=["sorter_name", "processing", "probe_type"]
    )
    # print(study)


def test_GroundTruthStudy(setup_module):
    study_folder = setup_module
    study = GroundTruthStudy(study_folder)
    print(study)

    study.run_sorters(verbose=True)

    print(study.sortings)

    print(study.comparisons)
    study.run_comparisons()
    print(study.comparisons)

    study.create_sorting_analyzer_gt(n_jobs=-1)

    study.compute_metrics()

    for key in study.cases:
        metrics = study.get_metrics(key)
        print(metrics)

    study.get_performance_by_unit()
    study.get_count_units()


if __name__ == "__main__":
    setup_module()
    test_GroundTruthStudy()
