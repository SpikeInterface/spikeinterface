from __future__ import annotations

import pytest
from pathlib import Path

from spikeinterface.core import generate_ground_truth_recording, start_sorting_result
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


def make_sorting_result(sparse=True, with_group=False):
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

    sorting_result = start_sorting_result(sorting=sorting, recording=recording, format="memory",  sparse=sparse)
    sorting_result.select_random_spikes()
    sorting_result.compute("waveforms")
    sorting_result.compute("templates")
    sorting_result.compute("noise_levels")
    sorting_result.compute("principal_components")
    sorting_result.compute("template_similarity")
    sorting_result.compute("quality_metrics", metric_names=["snr"])

    return sorting_result


@pytest.fixture(scope="module")
def sorting_result_dense_for_export():
    return make_sorting_result(sparse=False)


@pytest.fixture(scope="module")
def sorting_result_with_group_for_export():
    return make_sorting_result(sparse=False, with_group=True)


@pytest.fixture(scope="module")
def sorting_result_sparse_for_export():
    return make_sorting_result(sparse=True)


if __name__ == "__main__":
    sorting_result = make_sorting_result(sparse=False)
    print(sorting_result)
