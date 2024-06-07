import pytest

import shutil

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import create_sorting_analyzer
from spikeinterface.core import Templates

from spikeinterface.core.sortinganalyzer import _extension_children, _get_children_dependencies

import numpy as np


def get_sorting_analyzer(cache_folder, format="memory", sparse=True):
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
            unit_params=dict(
                alpha=(200.0, 500.0),
            )
        ),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
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


def _check_result_extension(sorting_analyzer, extension_name, cache_folder):
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
def test_ComputeRandomSpikes(format, sparse, create_cache_folder):
    cache_folder = create_cache_folder
    sorting_analyzer = get_sorting_analyzer(cache_folder, format=format, sparse=sparse)

    ext = sorting_analyzer.compute("random_spikes", max_spikes_per_unit=10, seed=2205)
    indices = ext.data["random_spikes_indices"]
    assert indices.size == 10 * sorting_analyzer.sorting.unit_ids.size

    _check_result_extension(sorting_analyzer, "random_spikes", cache_folder)
    sorting_analyzer.delete_extension("random_spikes")

    ext = sorting_analyzer.compute("random_spikes", method="all")
    indices = ext.data["random_spikes_indices"]
    assert indices.size == len(sorting_analyzer.sorting.to_spike_vector())

    _check_result_extension(sorting_analyzer, "random_spikes", cache_folder)


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeWaveforms(format, sparse, create_cache_folder):
    cache_folder = create_cache_folder
    sorting_analyzer = get_sorting_analyzer(cache_folder, format=format, sparse=sparse)

    job_kwargs = dict(n_jobs=2, chunk_duration="1s", progress_bar=True)
    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=50, seed=2205)
    ext = sorting_analyzer.compute("waveforms", **job_kwargs)
    wfs = ext.data["waveforms"]
    _check_result_extension(sorting_analyzer, "waveforms", cache_folder)


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeTemplates(format, sparse, create_cache_folder):
    cache_folder = create_cache_folder
    sorting_analyzer = get_sorting_analyzer(cache_folder, format=format, sparse=sparse)

    sorting_analyzer.compute("random_spikes", max_spikes_per_unit=50, seed=2205)

    job_kwargs = dict(n_jobs=2, chunk_duration="1s", progress_bar=True)

    with pytest.raises(ValueError):
        # This require "waveforms first and should trig an error
        sorting_analyzer.compute("templates", operators=["median"])

    with pytest.raises(ValueError):
        # This require "waveforms first and should trig an error
        sorting_analyzer.compute("templates", operators=[("percentile", 95.0)])

    ## without waveforms
    temp_ext = sorting_analyzer.compute("templates", operators=["average", "std"], **job_kwargs)

    fast_avg = temp_ext.get_templates(operator="average")
    fast_std = temp_ext.get_templates(operator="std")

    # with waveforms

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
    temp_ext = sorting_analyzer.get_extension("templates")
    templated_median = temp_ext.get_templates(operator="median")
    templated_per_5 = temp_ext.get_templates(operator="percentile", percentile=5.0)

    # they all should be in data
    data = sorting_analyzer.get_extension("templates").data
    for k in ["average", "std", "median", "pencentile_5.0", "pencentile_95.0"]:
        assert k in data.keys()
        assert data[k].shape[0] == sorting_analyzer.unit_ids.size
        assert data[k].shape[2] == sorting_analyzer.channel_ids.size
        assert np.any(data[k] > 0)

    if sorting_analyzer.sparsity is not None:
        sparsity_mask = sorting_analyzer.sparsity.mask
    else:
        sparsity_mask = np.ones((sorting_analyzer.unit_ids.size, sorting_analyzer.channel_ids.size), dtype=bool)
    for unit_index, unit_id in enumerate(sorting_analyzer.unit_ids):
        unit_mask = sparsity_mask[unit_index, :]
        assert np.allclose(
            fast_avg[unit_index][:, unit_mask], temp_ext.data["average"][unit_index][:, unit_mask], atol=0.01
        )
        assert np.allclose(fast_std[unit_index][:, unit_mask], temp_ext.data["std"][unit_index][:, unit_mask], atol=0.5)

    templates = temp_ext.get_templates(outputs="Templates")
    assert isinstance(templates, Templates)

    # import matplotlib.pyplot as plt
    # for unit_index, unit_id in enumerate(sorting_analyzer.unit_ids):
    #     fig, ax = plt.subplots()
    #     for k in data.keys():
    #         wf0 = data[k][unit_index, :, :]
    #         ax.plot(wf0.T.flatten(), label=k)
    #     ax.plot(fast_avg[unit_index].T.flatten(), label='fast_av', ls='--', color='k')
    #     ax.plot(fast_std[unit_index].T.flatten(), label='fast_std', ls='--', color='k')
    #     ax.legend()
    # plt.show()

    _check_result_extension(sorting_analyzer, "templates", cache_folder)


@pytest.mark.parametrize("format", ["memory", "binary_folder", "zarr"])
@pytest.mark.parametrize("sparse", [True, False])
def test_ComputeNoiseLevels(format, sparse, create_cache_folder):
    cache_folder = create_cache_folder
    sorting_analyzer = get_sorting_analyzer(cache_folder, format=format, sparse=sparse)

    sorting_analyzer.compute("noise_levels")
    print(sorting_analyzer)

    noise_levels = sorting_analyzer.get_extension("noise_levels").data["noise_levels"]
    assert noise_levels.shape[0] == sorting_analyzer.channel_ids.size


def test_get_children_dependencies():
    assert "waveforms" in _extension_children["random_spikes"]

    rs_children = _get_children_dependencies("random_spikes")
    assert "waveforms" in rs_children
    assert "templates" in rs_children

    assert rs_children.index("waveforms") < rs_children.index("templates")


def test_delete_on_recompute(create_cache_folder):
    cache_folder = create_cache_folder
    sorting_analyzer = get_sorting_analyzer(cache_folder, format="memory", sparse=False)
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("waveforms")
    sorting_analyzer.compute("templates")

    # re compute random_spikes should delete waveforms and templates
    sorting_analyzer.compute("random_spikes")
    assert sorting_analyzer.get_extension("templates") is None
    assert sorting_analyzer.get_extension("waveforms") is None


def test_compute_several(create_cache_folder):
    cache_folder = create_cache_folder
    sorting_analyzer = get_sorting_analyzer(cache_folder, format="memory", sparse=False)

    # should raise an error since waveforms depends on random_spikes, which isn't calculated
    with pytest.raises(AssertionError):
        sorting_analyzer.compute(["waveforms"])

    # check that waveforms are calculated
    sorting_analyzer.compute(["random_spikes", "waveforms"])
    waveform_data = sorting_analyzer.get_extension("waveforms").get_data()
    assert waveform_data is not None

    sorting_analyzer.delete_extension("waveforms")
    sorting_analyzer.delete_extension("random_spikes")

    # check that waveforms are calculated as before, even when parent is after child
    sorting_analyzer.compute(["waveforms", "random_spikes"])
    assert np.all(waveform_data == sorting_analyzer.get_extension("waveforms").get_data())


if __name__ == "__main__":

    test_ComputeWaveforms(format="memory", sparse=True)
    test_ComputeWaveforms(format="memory", sparse=False)
    test_ComputeWaveforms(format="binary_folder", sparse=True)
    test_ComputeWaveforms(format="binary_folder", sparse=False)
    test_ComputeWaveforms(format="zarr", sparse=True)
    test_ComputeWaveforms(format="zarr", sparse=False)
    test_ComputeRandomSpikes(format="memory", sparse=True)
    test_ComputeTemplates(format="memory", sparse=True)
    test_ComputeNoiseLevels(format="memory", sparse=False)

    test_get_children_dependencies()
    test_delete_on_recompute()
