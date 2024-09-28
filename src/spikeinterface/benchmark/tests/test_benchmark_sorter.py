import shutil
import pytest
from pathlib import Path

from spikeinterface import generate_ground_truth_recording
from spikeinterface.preprocessing import bandpass_filter
from spikeinterface.benchmark import SorterStudy


@pytest.fixture(scope="module")
def setup_module(tmp_path_factory):
    study_folder = tmp_path_factory.mktemp("sorter_study_folder")
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


def test_SorterStudy(setup_module):
    # job_kwargs = dict(n_jobs=2, chunk_duration="1s")

    study_folder = setup_module
    study = SorterStudy(study_folder)
    print(study)

    # # this run the sorters
    # study.run()

    # # this run comparisons
    # study.compute_results()
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

    


if __name__ == "__main__":
    study_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks" / "test_SorterStudy"
    create_a_study(study_folder)
    test_SorterStudy(study_folder)
