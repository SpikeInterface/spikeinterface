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
        durations=[30.0], sampling_frequency=16000.0, num_channels=10, num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting



def test_SortingResult_memory():
    recording, sorting = get_dataset()
    sortres = start_sorting_result(sorting, recording, format="memory", sparse=False, sparsity=None)
    _check_sorting_results(sortres, sorting)


def test_SortingResult_binary_folder():
    recording, sorting = get_dataset()

    folder = cache_folder / "test_SortingResult_binary_folder"
    if folder.exists():
        shutil.rmtree(folder)

    sortres = start_sorting_result(sorting, recording, format="binary_folder", folder=folder,  sparse=False, sparsity=None)
    sortres = load_sorting_result(folder, format="auto")

    _check_sorting_results(sortres, sorting)


def test_SortingResult_zarr():
    recording, sorting = get_dataset()

    folder = cache_folder / "test_SortingResult_zarr.zarr"
    if folder.exists():
        shutil.rmtree(folder)

    sortres = start_sorting_result(sorting, recording, format="zarr", folder=folder,  sparse=False, sparsity=None)
    # sortres = load_sorting_result(folder, format="auto")

    # _check_sorting_results(sortres, sorting)



def _check_sorting_results(sortres, original_sorting):

    print()
    print(sortres)

    register_result_extension(DummyResultExtension)

    assert "channel_ids" in sortres.rec_attributes
    assert "sampling_frequency" in sortres.rec_attributes
    assert "num_samples" in sortres.rec_attributes

    probe = sortres.get_probe()
    sparsity = sortres.sparsity

    # compute
    sortres.compute("dummy", param1=5.5)
    ext = sortres.get_extension("dummy")
    assert ext is not None
    assert ext.params["param1"] == 5.5
    # recompute
    sortres.compute("dummy", param1=5.5)
    # and delete
    sortres.delete_extension("dummy")
    ext = sortres.get_extension("dummy")
    assert ext is None


    # save to several format
    for format in ("memory", "binary_folder", ): # "zarr"
        if format != "memory":
            folder = cache_folder / f"test_SortingResult_save_as_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None

        # compute one extension to check the save
        sortres.compute("dummy")

        sortres2 = sortres.save_as(format=format, folder=folder)
        ext = sortres2.get_extension("dummy")
        assert ext is not None
        
        data = sortres2.get_extension("dummy").data
        assert "result_one" in data
        assert data["result_two"].size == original_sorting.to_spike_vector().size

    # select unit_ids to several format
    for format in ("memory", "binary_folder", ): # "zarr"
        if format != "memory":
            folder = cache_folder / f"test_SortingResult_select_units_with{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        # compute one extension to check the slice
        sortres.compute("dummy")
        keep_unit_ids = original_sorting.unit_ids[::2]
        sortres2 = sortres.select_units(unit_ids=keep_unit_ids, format=format, folder=folder)

        # check propagation of result data and correct sligin
        assert np.array_equal(keep_unit_ids, sortres2.unit_ids)
        data = sortres2.get_extension("dummy").data
        assert data["result_one"] == sortres.get_extension("dummy").data["result_one"]
        # unit 1, 3, ... should be removed
        assert np.all(~np.isin(data["result_two"], [-1, -3]))


class DummyResultExtension(ResultExtension):
    extension_name = "dummy"

    def _set_params(self, param0="yep", param1=1.2, param2=[1,2, 3.]):
        params = dict(param0=param0, param1=param1, param2=param2)
        params["more_option"] = "yep"
        return params
    
    def _run(self, **kwargs):
        # print("dummy run")
        self.data["result_one"] = "abcd"
        # the result two has the same size of the spike vector!!
        # and represent nothing (the trick is to use unit_index for testing slice)
        spikes = self.sorting_result.sorting.to_spike_vector()
        self.data["result_two"] = spikes["unit_index"] * -1
    
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
    # test_SortingResult_memory()
    # test_SortingResult_binary_folder()
    test_SortingResult_zarr()
    # test_extension()
