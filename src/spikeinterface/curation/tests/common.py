from __future__ import annotations

import pytest
from pathlib import Path

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer
from spikeinterface.qualitymetrics import compute_quality_metrics

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "curation"
else:
    cache_folder = Path("cache_folder") / "curation"


job_kwargs = dict(n_jobs=-1)


def make_sorting_analyzer(sparse=True):
    recording, sorting = generate_ground_truth_recording(
        durations=[300.0],
        sampling_frequency=30000.0,
        num_channels=4,
        num_units=5,
        generate_sorting_kwargs=dict(firing_rates=20.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )

    sorting_analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory", sparse=sparse)
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("noise_levels")
    # sorting_analyzer.compute("principal_components")
    # sorting_analyzer.compute("template_similarity")
    # sorting_analyzer.compute("quality_metrics", metric_names=["snr"])

    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_for_curation():
    return make_sorting_analyzer(sparse=True)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=False)
    print(sorting_analyzer)
