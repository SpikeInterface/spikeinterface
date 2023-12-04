import pytest
from pathlib import Path

from spikeinterface.core import generate_ground_truth_recording, extract_waveforms
from spikeinterface.postprocessing import (
    compute_spike_amplitudes,
    compute_template_similarity,
    compute_principal_components,
)
from spikeinterface.qualitymetrics import compute_quality_metrics

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "exporters"
else:
    cache_folder = Path("cache_folder") / "exporters"


def make_waveforms_extractor(sparse=True, with_group=False):
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=28000.0,
        num_channels=8,
        num_units=4,
        generate_probe_kwargs=dict(
            num_columns=2,
            xpitch=20,
            ypitch=20,
            contact_shapes="circle",
            contact_shape_params={"radius": 6},
        ),
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_level=5.0, strategy="on_the_fly"),
        seed=2205,
    )

    if with_group:
        recording.set_channel_groups([0, 0, 0, 0, 1, 1, 1, 1])
        sorting.set_property("group", [0, 0, 1, 1])

    we = extract_waveforms(recording=recording, sorting=sorting, folder=None, mode="memory", sparse=sparse)
    compute_principal_components(we)
    compute_spike_amplitudes(we)
    compute_template_similarity(we)
    compute_quality_metrics(we, metric_names=["snr"])

    return we


@pytest.fixture(scope="module")
def waveforms_extractor_dense_for_export():
    return make_waveforms_extractor(sparse=False)


@pytest.fixture(scope="module")
def waveforms_extractor_with_group_for_export():
    return make_waveforms_extractor(sparse=False, with_group=True)


@pytest.fixture(scope="module")
def waveforms_extractor_sparse_for_export():
    return make_waveforms_extractor(sparse=True)


if __name__ == "__main__":
    we = make_waveforms_extractor(sparse=False)
    print(we)
