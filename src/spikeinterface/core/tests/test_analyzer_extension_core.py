import pytest
from pathlib import Path

import shutil

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import create_sorting_analyzer

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def get_sorting_analyzer(format="memory", sparse=True):
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=20,
        num_units=5,
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

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format=format, folder=folder, sparse=sparse, sparsity=None
    )

    return sorting_analyzer


def _check_result_extension(sorting_analyzer, extension_name):
    # select unit_ids to several format
    for format in ("memory", "binary_folder", "zarr"):
        # for format in ("memory", ):
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_{extension_name}_select_units_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_{extension_name}_select_units_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None

        # check unit slice
        keep_unit_ids = sorting_analyzer.sorting.unit_ids[::2]
        sorting_analyzer2 = sorting_analyzer.select_units(unit_ids=keep_unit_ids, format=format, folder=folder)

        data = sorting_analyzer2.get_extension(extension_name).data
        # for k, arr in data.items():
        #     print(k, arr.shape)


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize(
    "sparse",
    [
        False,
    ],
)
def test_SelectRandomSpikes(format, sparse):
    sorting_analyzer = get_sorting_analyzer(format=format, sparse=sparse)

    ext = sorting_analyzer.compute("random_spikes", max_spikes_per_unit=10, seed=2205)
    indices = ext.data["random_spikes_indices"]
    assert indices.size == 10 * sorting_analyzer.sorting.unit_ids.size
    # print(indices)

    _check_result_extension(sorting_analyzer, "random_spikes")


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeWaveforms(format, sparse):
    sorting_analyzer = get_sorting_analyzer(format=format, sparse=sparse)

    job_kwargs = dict(n_jobs=2, chunk_duration="1s", progress_bar=True)
    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=50, seed=2205)
    ext = sorting_analyzer.compute("waveforms", **job_kwargs)
    wfs = ext.data["waveforms"]
    _check_result_extension(sorting_analyzer, "waveforms")


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeTemplates(format, sparse):
    sorting_analyzer = get_sorting_analyzer(format=format, sparse=sparse)

    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=20, seed=2205)

    with pytest.raises(AssertionError):
        # This require "waveforms first and should trig an error
        sorting_analyzer.compute("templates")

    job_kwargs = dict(n_jobs=2, chunk_duration="1s", progress_bar=True)
    sorting_analyzer.compute("waveforms", **job_kwargs)

    # compute some operators
    sorting_analyzer.compute(
        "templates",
        operators=[
            "average",
            "std",
            ("percentile", 95.0),
        ],
    )

    # ask for more operator later
    ext = sorting_analyzer.get_extension("templates")
    templated_median = ext.get_templates(operator="median")
    templated_per_5 = ext.get_templates(operator="percentile", percentile=5.0)

    # they all should be in data
    data = sorting_analyzer.get_extension("templates").data
    for k in ["average", "std", "median", "pencentile_5.0", "pencentile_95.0"]:
        assert k in data.keys()
        assert data[k].shape[0] == sorting_analyzer.unit_ids.size
        assert data[k].shape[2] == sorting_analyzer.channel_ids.size
        assert np.any(data[k] > 0)

    # import matplotlib.pyplot as plt
    # for unit_index, unit_id in enumerate(sorting_analyzer.unit_ids):
    #     fig, ax = plt.subplots()
    #     for k in data.keys():
    #         wf0 = data[k][unit_index, :, :]
    #         ax.plot(wf0.T.flatten(), label=k)
    #     ax.legend()
    # plt.show()

    _check_result_extension(sorting_analyzer, "templates")


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeFastTemplates(format, sparse):
    sorting_analyzer = get_sorting_analyzer(format=format, sparse=sparse)

    # TODO check this because this is not passing with n_jobs=2
    job_kwargs = dict(n_jobs=1, chunk_duration="1s", progress_bar=True)

    ms_before = 1.0
    ms_after = 2.5

    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=20, seed=2205)

    sorting_analyzer.compute("fast_templates", ms_before=ms_before, ms_after=ms_after, return_scaled=True, **job_kwargs)

    _check_result_extension(sorting_analyzer, "fast_templates")

    # compare ComputeTemplates with dense and ComputeFastTemplates: should give the same on "average"
    other_sorting_analyzer = get_sorting_analyzer(format=format, sparse=False)
    other_sorting_analyzer.compute("random_spikes", max_spikes_per_unit=20, seed=2205)
    other_sorting_analyzer.compute(
        "waveforms", ms_before=ms_before, ms_after=ms_after, return_scaled=True, **job_kwargs
    )
    other_sorting_analyzer.compute(
        "templates",
        operators=[
            "average",
        ],
    )

    templates0 = sorting_analyzer.get_extension("fast_templates").data["average"]
    templates1 = other_sorting_analyzer.get_extension("templates").data["average"]
    np.testing.assert_almost_equal(templates0, templates1)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # for unit_index, unit_id in enumerate(sorting_analyzer.unit_ids):
    #     wf0 = templates0[unit_index, :, :]
    #     ax.plot(wf0.T.flatten(), label=f"{unit_id}")
    #     wf1 = templates1[unit_index, :, :]
    #     ax.plot(wf1.T.flatten(), ls='--', color='k')
    # ax.legend()
    # plt.show()


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeNoiseLevels(format, sparse):
    sorting_analyzer = get_sorting_analyzer(format=format, sparse=sparse)

    sorting_analyzer.compute("noise_levels", return_scaled=True)
    print(sorting_analyzer)

    noise_levels = sorting_analyzer.get_extension("noise_levels").data["noise_levels"]
    assert noise_levels.shape[0] == sorting_analyzer.channel_ids.size


if __name__ == "__main__":

    test_SelectRandomSpikes(format="memory", sparse=True)

    test_ComputeWaveforms(format="memory", sparse=True)
    test_ComputeWaveforms(format="memory", sparse=False)
    test_ComputeWaveforms(format="binary_folder", sparse=True)
    test_ComputeWaveforms(format="binary_folder", sparse=False)
    test_ComputeWaveforms(format="zarr", sparse=True)
    test_ComputeWaveforms(format="zarr", sparse=False)

    test_ComputeTemplates(format="memory", sparse=True)
    test_ComputeTemplates(format="memory", sparse=False)
    test_ComputeTemplates(format="binary_folder", sparse=True)
    test_ComputeTemplates(format="zarr", sparse=True)

    test_ComputeFastTemplates(format="memory", sparse=True)

    test_ComputeNoiseLevels(format="memory", sparse=False)
