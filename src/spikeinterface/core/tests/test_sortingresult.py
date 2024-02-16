import pytest
from pathlib import Path

import shutil

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import SortingResult, start_sorting_result, load_sorting_result
from spikeinterface.core.sortingresult import register_result_extension, ResultExtension

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting


def test_SortingResult_memory():
    recording, sorting = get_dataset()
    sorting_result = start_sorting_result(sorting, recording, format="memory", sparse=False, sparsity=None)
    _check_sorting_results(sorting_result, sorting)

    sorting_result = start_sorting_result(sorting, recording, format="memory", sparse=True, sparsity=None)
    _check_sorting_results(sorting_result, sorting)


def test_SortingResult_binary_folder():
    recording, sorting = get_dataset()

    folder = cache_folder / "test_SortingResult_binary_folder"
    if folder.exists():
        shutil.rmtree(folder)

    sorting_result = start_sorting_result(
        sorting, recording, format="binary_folder", folder=folder, sparse=False, sparsity=None
    )
    sorting_result = load_sorting_result(folder, format="auto")
    _check_sorting_results(sorting_result, sorting)


def test_SortingResult_zarr():
    recording, sorting = get_dataset()

    folder = cache_folder / "test_SortingResult_zarr.zarr"
    if folder.exists():
        shutil.rmtree(folder)

    sorting_result = start_sorting_result(sorting, recording, format="zarr", folder=folder, sparse=False, sparsity=None)
    sorting_result = load_sorting_result(folder, format="auto")
    _check_sorting_results(sorting_result, sorting)


def _check_sorting_results(sorting_result, original_sorting):

    print()
    print(sorting_result)

    register_result_extension(DummyResultExtension)

    assert "channel_ids" in sorting_result.rec_attributes
    assert "sampling_frequency" in sorting_result.rec_attributes
    assert "num_samples" in sorting_result.rec_attributes

    probe = sorting_result.get_probe()
    sparsity = sorting_result.sparsity

    # compute
    sorting_result.compute("dummy", param1=5.5)
    # equivalent
    compute_dummy(sorting_result, param1=5.5)
    ext = sorting_result.get_extension("dummy")
    assert ext is not None
    assert ext.params["param1"] == 5.5
    print(sorting_result)
    # recompute
    sorting_result.compute("dummy", param1=5.5)
    # and delete
    sorting_result.delete_extension("dummy")
    ext = sorting_result.get_extension("dummy")
    assert ext is None

    assert sorting_result.has_recording()

    if sorting_result.random_spikes_indices is None:
        sorting_result.select_random_spikes(max_spikes_per_unit=10, seed=2205)
        assert sorting_result.random_spikes_indices is not None
        assert sorting_result.random_spikes_indices.size == 10 * sorting_result.sorting.unit_ids.size

    # save to several format
    for format in ("memory", "binary_folder", "zarr"):
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingResult_save_as_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingResult_save_as_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None

        # compute one extension to check the save
        sorting_result.compute("dummy")

        sorting_result2 = sorting_result.save_as(format=format, folder=folder)
        ext = sorting_result2.get_extension("dummy")
        assert ext is not None

        data = sorting_result2.get_extension("dummy").data
        assert "result_one" in data
        assert data["result_two"].size == original_sorting.to_spike_vector().size

    # select unit_ids to several format
    for format in ("memory", "binary_folder", "zarr"):
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingResult_select_units_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingResult_select_units_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        # compute one extension to check the slice
        sorting_result.compute("dummy")
        keep_unit_ids = original_sorting.unit_ids[::2]
        sorting_result2 = sorting_result.select_units(unit_ids=keep_unit_ids, format=format, folder=folder)

        # check that random_spikes_indices are remmaped
        assert sorting_result2.random_spikes_indices is not None
        some_spikes = sorting_result2.sorting.to_spike_vector()[sorting_result2.random_spikes_indices]
        assert np.array_equal(np.unique(some_spikes["unit_index"]), np.arange(keep_unit_ids.size))

        # check propagation of result data and correct sligin
        assert np.array_equal(keep_unit_ids, sorting_result2.unit_ids)
        data = sorting_result2.get_extension("dummy").data
        assert data["result_one"] == sorting_result.get_extension("dummy").data["result_one"]
        # unit 1, 3, ... should be removed
        assert np.all(~np.isin(data["result_two"], [1, 3]))


class DummyResultExtension(ResultExtension):
    extension_name = "dummy"
    depend_on = []
    need_recording = False
    use_nodepipeline = False

    def _set_params(self, param0="yep", param1=1.2, param2=[1, 2, 3.0]):
        params = dict(param0=param0, param1=param1, param2=param2)
        params["more_option"] = "yep"
        return params

    def _run(self, **kwargs):
        # print("dummy run")
        self.data["result_one"] = "abcd"
        # the result two has the same size of the spike vector!!
        # and represent nothing (the trick is to use unit_index for testing slice)
        spikes = self.sorting_result.sorting.to_spike_vector()
        self.data["result_two"] = spikes["unit_index"].copy()

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_result.unit_ids, unit_ids))

        spikes = self.sorting_result.sorting.to_spike_vector()
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)
        # here the first key do not depend on unit_id
        # but the second need to be sliced!!
        new_data = dict()
        new_data["result_one"] = self.data["result_one"]
        new_data["result_two"] = self.data["result_two"][keep_spike_mask]

        return new_data

    def _get_data(self):
        return self.data["result_one"]


compute_dummy = DummyResultExtension.function_factory()


class DummyResultExtension2(ResultExtension):
    extension_name = "dummy"


def test_extension():
    register_result_extension(DummyResultExtension)
    # can be register twice without error
    register_result_extension(DummyResultExtension)

    # other extension with same name should trigger an error
    with pytest.raises(AssertionError):
        register_result_extension(DummyResultExtension2)


if __name__ == "__main__":
    test_SortingResult_memory()
    test_SortingResult_binary_folder()
    test_SortingResult_zarr()
    test_extension()
