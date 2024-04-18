import pytest
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from spikeinterface.core import (
    NumpySorting,
    synthetize_spike_train_bad_isi,
    add_synchrony_to_sorting,
    generate_ground_truth_recording,
    create_sorting_analyzer,
)

# from spikeinterface.extractors.toy_example import toy_example
from spikeinterface.qualitymetrics.utils import create_ground_truth_pc_distributions

from spikeinterface.qualitymetrics import (
    calculate_pc_metrics,
    nearest_neighbors_isolation,
    nearest_neighbors_noise_overlap,
)


job_kwargs = dict(n_jobs=2, progress_bar=True, chunk_duration="1s")


def _sorting_analyzer_simple():
    recording, sorting = generate_ground_truth_recording(
        durations=[
            50.0,
        ],
        sampling_frequency=30_000.0,
        num_channels=6,
        num_units=10,
        generate_sorting_kwargs=dict(firing_rates=6.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=True)

    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=300, seed=2205)
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates", operators=["average", "std", "median"])
    sorting_analyzer.compute("principal_components", n_components=5, mode="by_channel_local", **job_kwargs)
    sorting_analyzer.compute("spike_amplitudes", **job_kwargs)

    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_simple():
    return _sorting_analyzer_simple()


def test_calculate_pc_metrics(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    res1 = calculate_pc_metrics(sorting_analyzer, n_jobs=1, progress_bar=True)
    res1 = pd.DataFrame(res1)

    res2 = calculate_pc_metrics(sorting_analyzer, n_jobs=2, progress_bar=True)
    res2 = pd.DataFrame(res2)

    for k in res1.columns:
        mask = ~np.isnan(res1[k].values)
        if np.any(mask):
            assert np.array_equal(res1[k].values[mask], res2[k].values[mask])


def test_nearest_neighbors_isolation(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    this_unit_id = sorting_analyzer.unit_ids[0]
    nearest_neighbors_isolation(sorting_analyzer, this_unit_id)


def test_nearest_neighbors_noise_overlap(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    this_unit_id = sorting_analyzer.unit_ids[0]
    nearest_neighbors_noise_overlap(sorting_analyzer, this_unit_id)


if __name__ == "__main__":
    sorting_analyzer = _sorting_analyzer_simple()
    test_calculate_pc_metrics(sorting_analyzer)
    test_nearest_neighbors_isolation(sorting_analyzer)
    test_nearest_neighbors_noise_overlap(sorting_analyzer)
