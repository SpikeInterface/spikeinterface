import shutil
import pytest
from pathlib import Path

from spikeinterface import generate_ground_truth_recording
from spikeinterface.preprocessing import bandpass_filter
from spikeinterface.benchmark import SorterStudyWithoutGroundTruth


@pytest.fixture(scope="module")
def create_simple_study_no_gt(tmp_path_factory):
    study_folder = tmp_path_factory.mktemp("sorter_study_folder")
    if study_folder.is_dir():
        shutil.rmtree(study_folder)
    _create_simple_study_no_gt(study_folder)
    return study_folder


def _create_simple_study_no_gt(study_folder):
    rec1, __annotations__ = generate_ground_truth_recording(num_channels=32, durations=[300.0], seed=2205)

    datasets = {
        "toy_probe32": (rec1, None),
    }

    # cases can also be generated via simple loops
    cases = {
        #
        "tdc2": {
            "label": "tridesclous2",
            "dataset": "toy_probe32",
            "params": {
                "sorter_name": "tridesclous2",
            },
        },
        "sc2": {
            "label": "spykingcircus2",
            "dataset": "toy_probe32",
            "params": {
                "sorter_name": "spykingcircus2",
            },
        },
    }

    study = SorterStudyWithoutGroundTruth.create(study_folder, datasets=datasets, cases=cases)
    # print(study)


@pytest.mark.skip()
def test_SorterStudyWithoutGroundTruth(create_simple_study):
    # job_kwargs = dict(n_jobs=2, chunk_duration="1s")

    study_folder = create_simple_study
    study = SorterStudyWithoutGroundTruth(study_folder)
    print(study)

    # # this run the sorters
    study.run()

    # # this run comparisons
    study.compute_results()
    print(study)


if __name__ == "__main__":
    study_folder_simple = (
        Path(__file__).resolve().parents[4] / "cache_folder" / "benchmarks" / "test_SorterStudyWithoutGroundTruth"
    )
    print(study_folder_simple)
    if study_folder_simple.exists():
        shutil.rmtree(study_folder_simple)
    _create_simple_study_no_gt(study_folder_simple)
    test_SorterStudyWithoutGroundTruth(study_folder_simple)
