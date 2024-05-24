from __future__ import annotations

import pytest
from pathlib import Path

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer, compute_sparsity

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "exporters"
else:
    cache_folder = Path("cache_folder") / "exporters"


def make_sorting_analyzer(sparse=True, with_group=False):
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
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )

    if with_group:
        recording.set_channel_groups([0, 0, 0, 0, 1, 1, 1, 1])
        sorting.set_property("group", [0, 0, 1, 1])

        sorting_analyzer_unused = create_sorting_analyzer(
            sorting=sorting, recording=recording, format="memory", sparse=False, sparsity=None
        )
        sparsity_group = compute_sparsity(sorting_analyzer_unused, method="by_property", by_property="group")

        sorting_analyzer = create_sorting_analyzer(
            sorting=sorting, recording=recording, format="memory", sparse=False, sparsity=sparsity_group
        )
    else:
        sorting_analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory", sparse=sparse)

    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("waveforms")
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("principal_components")
    sorting_analyzer.compute("template_similarity")
    sorting_analyzer.compute("quality_metrics", metric_names=["snr"])

    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_dense_for_export():
    return make_sorting_analyzer(sparse=False)


@pytest.fixture(scope="module")
def sorting_analyzer_with_group_for_export():
    return make_sorting_analyzer(sparse=False, with_group=True)


@pytest.fixture(scope="module")
def sorting_analyzer_sparse_for_export():
    return make_sorting_analyzer(sparse=True)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=False)
    print(sorting_analyzer)
