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


def get_sorting_result(format="memory"):
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0], sampling_frequency=16000.0, num_channels=10, num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    if format == "memory":
        folder = None
    elif format == "binary_folder":
        folder = cache_folder / f"test_ComputeWaveforms_{format}"
    elif format == "zarr":
        folder = cache_folder / f"test_ComputeWaveforms.zarr"
    if folder and folder.exists():
        shutil.rmtree(folder)
    
    sortres = start_sorting_result(sorting, recording, format=format, folder=folder,  sparse=False, sparsity=None)

    return sortres


def _check_result_extension(sortres, extension_name):


    # select unit_ids to several format
    # for format in ("memory", "binary_folder", "zarr"):
    for format in ("memory", ):
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingResult_{extension_name}_select_units_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingResult_{extension_name}_select_units_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None

        # check unit slice
        keep_unit_ids = sortres.sorting.unit_ids[::2]
        sortres2 = sortres.select_units(unit_ids=keep_unit_ids, format=format, folder=folder)

        data = sortres2.get_extension(extension_name).data
        # for k, arr in data.items():
        #     print(k, arr.shape)


def test_ComputeWaveforms(format="memory"):
    sortres = get_sorting_result(format=format)

    sortres.select_random_spikes(max_spikes_per_unit=50, seed=2205)
    ext = sortres.compute("waveforms")
    wfs = ext.data["waveforms"]
    _check_result_extension(sortres, "waveforms")


def test_ComputeTemplates(format="memory"):
    sortres = get_sorting_result(format=format)

    sortres.select_random_spikes(max_spikes_per_unit=20, seed=2205)
    
    with pytest.raises(AssertionError):
        # This require "waveforms first and should trig an error
        sortres.compute("templates")
    
    sortres.compute("waveforms")
    waveforms = sortres.get_extension("waveforms").data["waveforms"]
    sortres.compute("templates", operators=["average", "std", "median", ("percentile", 5.), ("percentile", 95.),])


    data = sortres.get_extension("templates").data
    for k in ['average', 'std', 'median', 'pencentile_5.0', 'pencentile_95.0']:
        assert k in data.keys()
        assert data[k].shape[0] == sortres.sorting.unit_ids.size
        assert data[k].shape[1] == waveforms.shape[1]


    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # unit_index = 2
    # for k in data.keys():
    #     wf0 = data[k][unit_index, :, :]
    #     ax.plot(wf0.T.flatten(), label=k)
    # ax.legend()
    # plt.show()

    _check_result_extension(sortres, "templates")

if __name__ == '__main__':
    # test_ComputeWaveforms(format="memory")
    # test_ComputeWaveforms(format="binary_folder")
    # test_ComputeWaveforms(format="zarr")
    test_ComputeTemplates(format="memory")
    # test_ComputeTemplates(format="binary_folder")
    # test_ComputeTemplates(format="zarr")
