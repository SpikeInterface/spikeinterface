import pytest

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer, aggregate_units
from spikeinterface.core.generate import inject_some_split_units
from spikeinterface.curation import train_model
from pathlib import Path

extensions = [
    "noise_levels",
    "random_spikes",
    "waveforms",
    "templates",
    "unit_locations",
    "spike_amplitudes",
    "spike_locations",
    "correlograms",
    "template_similarity",
]


def make_sorting_analyzer(sparse=True, num_units=5, durations=[300.0]):
    recording, sorting = generate_ground_truth_recording(
        durations=durations,
        sampling_frequency=30000.0,
        num_channels=4,
        num_units=num_units,
        generate_sorting_kwargs=dict(firing_rates=20.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )

    channel_ids_as_integers = [id for id in range(recording.get_num_channels())]
    unit_ids_as_integers = [id for id in range(sorting.get_num_units())]
    recording = recording.rename_channels(new_channel_ids=channel_ids_as_integers)
    sorting = sorting.rename_units(new_unit_ids=unit_ids_as_integers)

    sorting_analyzer = create_sorting_analyzer(
        sorting=sorting, recording=recording, format="memory", sparse=sparse, n_jobs=-1
    )
    sorting_analyzer.compute(extensions, n_jobs=-1)

    return sorting_analyzer


def make_sorting_analyzer_with_splits(sorting_analyzer, num_unit_splitted=1, num_split=2):
    sorting = sorting_analyzer.sorting

    split_ids = sorting.unit_ids[:num_unit_splitted]
    sorting_with_split, other_ids = inject_some_split_units(
        sorting,
        split_ids=split_ids,
        num_split=num_split,
        output_ids=True,
        seed=42,
    )

    sorting_analyzer_with_splits = create_sorting_analyzer(
        sorting=sorting_with_split, recording=sorting_analyzer.recording, format="memory", sparse=True, n_jobs=-1
    )
    sorting_analyzer_with_splits.compute(extensions, n_jobs=-1)

    return sorting_analyzer_with_splits, num_unit_splitted, other_ids


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
    analyzer = create_sorting_analyzer(sorting=both_sortings, recording=recording, n_jobs=-1)
    analyzer.compute(["random_spikes", "noise_levels", "templates"], n_jobs=-1)
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


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=False)
    print(sorting_analyzer)
