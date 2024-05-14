import pytest
from pathlib import Path

import shutil

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core import (
    create_sorting_analyzer,
    load_sorting_analyzer,
    get_available_analyzer_extensions,
    get_default_analyzer_extension_params,
)
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension

import numpy as np


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    return recording, sorting


def test_SortingAnalyzer_memory(tmp_path):
    recording, sorting = get_dataset()
    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=False, sparsity=None)
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=True, sparsity=None)
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="memory", sparse=False, return_scaled=True, sparsity=None
    )
    assert sorting_analyzer.return_scaled
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="memory", sparse=False, return_scaled=False, sparsity=None
    )
    assert not sorting_analyzer.return_scaled


def test_SortingAnalyzer_binary_folder(tmp_path):
    recording, sorting = get_dataset()

    folder = tmp_path / "test_SortingAnalyzer_binary_folder"
    if folder.exists():
        shutil.rmtree(folder)

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="binary_folder", folder=folder, sparse=False, sparsity=None
    )
    sorting_analyzer = load_sorting_analyzer(folder, format="auto")
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    folder = tmp_path / "test_SortingAnalyzer_binary_folder"
    if folder.exists():
        shutil.rmtree(folder)

    sorting_analyzer = create_sorting_analyzer(
        sorting,
        recording,
        format="binary_folder",
        folder=folder,
        sparse=False,
        sparsity=None,
        return_scaled=False,
    )
    assert not sorting_analyzer.return_scaled
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)


def test_SortingAnalyzer_zarr(tmp_path):
    recording, sorting = get_dataset()

    folder = tmp_path / "test_SortingAnalyzer_zarr.zarr"
    if folder.exists():
        shutil.rmtree(folder)

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="zarr", folder=folder, sparse=False, sparsity=None
    )
    sorting_analyzer = load_sorting_analyzer(folder, format="auto")
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    folder = tmp_path / "test_SortingAnalyzer_zarr.zarr"
    if folder.exists():
        shutil.rmtree(folder)
    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="zarr", folder=folder, sparse=False, sparsity=None, return_scaled=False
    )


def _check_sorting_analyzers(sorting_analyzer, original_sorting, cache_folder):

    print()
    print(sorting_analyzer)

    register_result_extension(DummyAnalyzerExtension)

    assert "channel_ids" in sorting_analyzer.rec_attributes
    assert "sampling_frequency" in sorting_analyzer.rec_attributes
    assert "num_samples" in sorting_analyzer.rec_attributes

    probe = sorting_analyzer.get_probe()
    sparsity = sorting_analyzer.sparsity

    # compute
    sorting_analyzer.compute("dummy", param1=5.5)
    # equivalent
    compute_dummy(sorting_analyzer=sorting_analyzer, param1=5.5)
    ext = sorting_analyzer.get_extension("dummy")
    assert ext is not None
    assert ext.params["param1"] == 5.5
    print(sorting_analyzer)
    # recompute
    sorting_analyzer.compute("dummy", param1=5.5)
    # and delete
    sorting_analyzer.delete_extension("dummy")
    ext = sorting_analyzer.get_extension("dummy")
    assert ext is None

    assert sorting_analyzer.has_recording()

    # save to several format
    for format in ("memory", "binary_folder", "zarr"):
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_save_as_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_save_as_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None

        # compute one extension to check the save
        sorting_analyzer.compute("dummy")

        sorting_analyzer2 = sorting_analyzer.save_as(format=format, folder=folder)
        ext = sorting_analyzer2.get_extension("dummy")
        assert ext is not None

        data = sorting_analyzer2.get_extension("dummy").data
        assert "result_one" in data
        assert data["result_two"].size == original_sorting.to_spike_vector().size

        assert sorting_analyzer2.return_scaled == sorting_analyzer.return_scaled

    # select unit_ids to several format
    for format in ("memory", "binary_folder", "zarr"):
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_select_units_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_select_units_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        # compute one extension to check the slice
        sorting_analyzer.compute("dummy")
        keep_unit_ids = original_sorting.unit_ids[::2]
        sorting_analyzer2 = sorting_analyzer.select_units(unit_ids=keep_unit_ids, format=format, folder=folder)

        # check propagation of result data and correct sligin
        assert np.array_equal(keep_unit_ids, sorting_analyzer2.unit_ids)
        data = sorting_analyzer2.get_extension("dummy").data
        assert data["result_one"] == sorting_analyzer.get_extension("dummy").data["result_one"]
        # unit 1, 3, ... should be removed
        assert np.all(~np.isin(data["result_two"], [1, 3]))

    # test compute with extension-specific params
    sorting_analyzer.compute(["dummy"], extension_params={"dummy": {"param1": 5.5}})
    dummy_ext = sorting_analyzer.get_extension("dummy")
    assert dummy_ext.params["param1"] == 5.5


def test_extension_params():
    from spikeinterface.core.sortinganalyzer import _builtin_extensions

    computable_extension = get_available_analyzer_extensions()

    for ext, mod in _builtin_extensions.items():
        assert ext in computable_extension
        if mod == "spikeinterface.core":
            default_params = get_default_analyzer_extension_params(ext)
            print(ext, default_params)
        else:
            try:
                default_params = get_default_analyzer_extension_params(ext)
                print(ext, default_params)
            except:
                print(f"Failed to import {ext}")


class DummyAnalyzerExtension(AnalyzerExtension):
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
        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        self.data["result_two"] = spikes["unit_index"].copy()

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)
        # here the first key do not depend on unit_id
        # but the second need to be sliced!!
        new_data = dict()
        new_data["result_one"] = self.data["result_one"]
        new_data["result_two"] = self.data["result_two"][keep_spike_mask]

        return new_data

    def _get_data(self):
        return self.data["result_one"]


compute_dummy = DummyAnalyzerExtension.function_factory()


class DummyAnalyzerExtension2(AnalyzerExtension):
    extension_name = "dummy"


def test_extension():
    register_result_extension(DummyAnalyzerExtension)
    # can be register twice without error
    register_result_extension(DummyAnalyzerExtension)

    # other extension with same name should trigger an error
    with pytest.raises(AssertionError):
        register_result_extension(DummyAnalyzerExtension2)


if __name__ == "__main__":
    tmp_path = Path("test_SortingAnalyzer")
    test_SortingAnalyzer_memory(tmp_path)
    test_SortingAnalyzer_binary_folder(tmp_path)
    test_SortingAnalyzer_zarr(tmp_path)
    test_extension()
    test_extension_params()
