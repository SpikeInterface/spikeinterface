import pytest
from pathlib import Path

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer, aggregate_units
from spikeinterface.curation import train_model
from spikeinterface.curation.tests.common import make_sorting_analyzer, make_sorting_analyzer_with_splits


@pytest.fixture(scope="module")
def sorting_analyzer_for_curation():
    return make_sorting_analyzer(sparse=True)


@pytest.fixture(scope="module")
def sorting_analyzer_for_unitrefine_curation():
    """Makes an analyzer whose first 10 units are good normal units, and 10 which are noise. We make them
    noise by using a spike trains which are uncorrelated with the recording for `sorting2`."""

    recording, sorting_1 = generate_ground_truth_recording(num_channels=4, seed=1, num_units=6)
    _, sorting_2 = generate_ground_truth_recording(num_channels=4, seed=2, num_units=6)
    both_sortings = aggregate_units([sorting_1, sorting_2])
    analyzer = create_sorting_analyzer(sorting=both_sortings, recording=recording)
    analyzer.compute(["random_spikes", "noise_levels", "templates"])
    return analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_multi_segment_for_curation():
    return make_sorting_analyzer(sparse=True, durations=[50.0, 30.0])


@pytest.fixture(scope="module")
def sorting_analyzer_with_splits():
    sorting_analyzer = make_sorting_analyzer(sparse=True, durations=[50.0])
    return make_sorting_analyzer_with_splits(sorting_analyzer)


@pytest.fixture(scope="module")
def trained_pipeline_path(sorting_analyzer_for_unitrefine_curation):
    """
    Makes a model saved at "./trained_pipeline" which will be used by other tests in the module.
    If the model already exists, this function does nothing.
    """
    trained_model_folder = Path(__file__).parent / Path("trained_pipeline")
    if trained_model_folder.is_dir():
        yield trained_model_folder
    else:
        analyzer = sorting_analyzer_for_unitrefine_curation
        analyzer.compute(
            {
                "quality_metrics": {"metric_names": ["snr"]},
                "template_metrics": {"metric_names": ["half_width", "peak_to_trough_duration", "number_of_peaks"]},
            }
        )
        train_model(
            analyzers=[analyzer] * 5,
            folder=trained_model_folder,
            labels=[[1] * 6 + [0] * 6] * 5,
            imputation_strategies=["median"],
            scaling_techniques=["standard_scaler"],
            classifiers=["RandomForestClassifier"],
            overwrite=True,
            search_kwargs={"cv": 3, "scoring": "balanced_accuracy", "n_iter": 2},
        )
        yield trained_model_folder
