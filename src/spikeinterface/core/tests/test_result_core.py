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


def get_sorting_result(format="memory", sparse=True):
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0], sampling_frequency=16000.0, num_channels=20, num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        generate_unit_locations_kwargs=dict(
            margin_um=5.0,
            minimum_z=5.0,
            maximum_z=20.0,
        ),
        generate_templates_kwargs=dict(
            unit_params_range=dict(
                alpha=(9_000.0, 12_000.0),
            )
        ),
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2406,
    )
    if format == "memory":
        folder = None
    elif format == "binary_folder":
        folder = cache_folder / f"test_ComputeWaveforms_{format}"
    elif format == "zarr":
        folder = cache_folder / f"test_ComputeWaveforms.zarr"
    if folder and folder.exists():
        shutil.rmtree(folder)
    
    sortres = start_sorting_result(sorting, recording, format=format, folder=folder, sparse=sparse, sparsity=None)

    return sortres


def _check_result_extension(sortres, extension_name):
    # select unit_ids to several format
    for format in ("memory", "binary_folder", "zarr"):
    # for format in ("memory", ):
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


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeWaveforms(format, sparse):
    sortres = get_sorting_result(format=format, sparse=sparse)

    sortres.select_random_spikes(max_spikes_per_unit=50, seed=2205)
    ext = sortres.compute("waveforms")
    wfs = ext.data["waveforms"]
    _check_result_extension(sortres, "waveforms")


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeTemplates(format, sparse):
    sortres = get_sorting_result(format=format, sparse=sparse)

    sortres.select_random_spikes(max_spikes_per_unit=20, seed=2205)
    
    with pytest.raises(AssertionError):
        # This require "waveforms first and should trig an error
        sortres.compute("templates")
    
    sortres.compute("waveforms")
    sortres.compute("templates", operators=["average", "std", "median", ("percentile", 5.), ("percentile", 95.),])


    data = sortres.get_extension("templates").data
    for k in ['average', 'std', 'median', 'pencentile_5.0', 'pencentile_95.0']:
        assert k in data.keys()
        assert data[k].shape[0] == sortres.unit_ids.size
        assert data[k].shape[2] == sortres.channel_ids.size
        assert np.any(data[k] > 0)

    import matplotlib.pyplot as plt
    for unit_index, unit_id in enumerate(sortres.unit_ids):
        fig, ax = plt.subplots()
        for k in data.keys():
            wf0 = data[k][unit_index, :, :]
            ax.plot(wf0.T.flatten(), label=k)
        ax.legend()
    # plt.show()

    _check_result_extension(sortres, "templates")


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeFastTemplates(format, sparse):
    sortres = get_sorting_result(format=format, sparse=sparse)

    ms_before=1.0
    ms_after=2.5

    sortres.select_random_spikes(max_spikes_per_unit=20, seed=2205)
    sortres.compute("fast_templates", ms_before=ms_before, ms_after=ms_after, return_scaled=True)

    _check_result_extension(sortres, "fast_templates")

    # compare ComputeTemplates with dense and ComputeFastTemplates: should give the same on "average"
    other_sortres = get_sorting_result(format=format, sparse=False)
    other_sortres.select_random_spikes(max_spikes_per_unit=20, seed=2205)
    other_sortres.compute("waveforms", ms_before=ms_before, ms_after=ms_after, return_scaled=True)
    other_sortres.compute("templates", operators=["average",])

    templates0 = sortres.get_extension("fast_templates").data["average"]
    templates1 = other_sortres.get_extension("templates").data["average"]
    np.testing.assert_almost_equal(templates0, templates1)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # for unit_index, unit_id in enumerate(sortres.unit_ids):
    #     wf0 = templates0[unit_index, :, :]
    #     ax.plot(wf0.T.flatten(), label=f"{unit_id}")
    #     wf1 = templates1[unit_index, :, :]
    #     ax.plot(wf1.T.flatten(), ls='--', color='k')
    # ax.legend()
    # plt.show()



if __name__ == '__main__':
    # test_ComputeWaveforms(format="memory", sparse=True)
    # test_ComputeWaveforms(format="memory", sparse=False)
    # test_ComputeWaveforms(format="binary_folder", sparse=True)
    # test_ComputeWaveforms(format="binary_folder", sparse=False)
    # test_ComputeWaveforms(format="zarr", sparse=True)
    # test_ComputeWaveforms(format="zarr", sparse=False)

    test_ComputeTemplates(format="memory", sparse=True)
    test_ComputeTemplates(format="memory", sparse=False)
    test_ComputeTemplates(format="binary_folder", sparse=True)
    test_ComputeTemplates(format="zarr", sparse=True)

    # test_ComputeFastTemplates(format="memory", sparse=True)
