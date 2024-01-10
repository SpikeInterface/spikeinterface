import pytest
from pathlib import Path

import shutil

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import SortingResult
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
    sortres = SortingResult.create(sorting, recording, format="memory", sparsity=None)
    print(sortres.rec_attributes.keys())
    _check_sorting_results(sortres)

    # save to binrary_folder
    folder = cache_folder / "test_SortingResult_saved_binary_folder"
    if folder.exists():
        shutil.rmtree(folder)

    sortres2 = sortres.save_as(folder, format="binary_folder")
    _check_sorting_results(sortres2)

    # save to zarr
    # folder = cache_folder / "test_SortingResult_saved_zarr.zarr"
    # if folder.exists():
    #     shutil.rmtree(folder)
    # sortres2 = sortres.save_as(folder, format="zarr")
    # _check_sorting_results(sortres2)



def test_SortingResult_folder():
    recording, sorting = get_dataset()

    folder = cache_folder / "test_SortingResult_folder"
    if folder.exists():
        shutil.rmtree(folder)

    sortres = SortingResult.create(sorting, recording, format="binary_folder", folder=folder, sparsity=None)
    sortres = SortingResult.load(folder)
    
    print(sortres.folder)

    _check_sorting_results(sortres)

def _check_sorting_results(sortres):
    register_result_extension(DummyResultExtension)

    print()
    print(sortres)
    print(sortres.sampling_frequency)
    print(sortres.channel_ids)
    print(sortres.unit_ids)
    print(sortres.get_probe())
    print(sortres.sparsity)

    sortres.compute("dummy", param1=5.5)
    ext = sortres.get_extension("dummy")
    assert ext is not None
    assert ext._params["param1"] == 5.5
    sortres.compute("dummy", param1=5.5)

    sortres.delete_extension("dummy")
    ext = sortres.get_extension("dummy")
    assert ext is None


class DummyResultExtension(ResultExtension):
    extension_name = "dummy"

    def _set_params(self, param0="yep", param1=1.2, param2=[1,2, 3.]):
        params = dict(param0=param0, param1=param1, param2=param2)
        params["more_option"] = "yep"
        return params
    
    def _run(self, **kwargs):
        # print("dummy run")
        self._data["result_one"] = "abcd"
        self._data["result_two"] = np.zeros(3)


class DummyResultExtension2(ResultExtension):
    extension_name = "dummy"


def test_extension():
    register_result_extension(DummyResultExtension)
    # can be register twice
    register_result_extension(DummyResultExtension)

    # same name should trigger an error
    with pytest.raises(AssertionError):
        register_result_extension(DummyResultExtension2)


if __name__ == "__main__":
    test_SortingResult_memory()

    # test_SortingResult_folder()

    # test_extension()