import pytest
from pathlib import Path

import shutil

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import start_sorting_result

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0], sampling_frequency=16000.0, num_channels=10, num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting




def test_ComputeWaveforms(format="memory"):

    if format == "memory":
        folder = None
    elif format == "binary_folder":
        folder = cache_folder / f"test_ComputeWaveforms_{format}"
    elif format == "zarr":
        folder = cache_folder / f"test_ComputeWaveforms.zarr"
    if folder and folder.exists():
        shutil.rmtree(folder)
        
    recording, sorting = get_dataset()
    sortres = start_sorting_result(sorting, recording, format=format, folder=folder, sparse=False, sparsity=None)
    print(sortres)

    sortres.select_random_spikes(max_spikes_per_unit=50, seed=2205)
    ext = sortres.compute("waveforms")
    wfs = ext.data["waveforms"]

    print(wfs.shape)
    print(sortres)


def test_ComputTemplates():
    pass

if __name__ == '__main__':
    # test_ComputeWaveforms(format="memory")
    test_ComputeWaveforms(format="binary_folder")
    # test_ComputeWaveforms(format="zarr")
    # test_ComputTemplates()