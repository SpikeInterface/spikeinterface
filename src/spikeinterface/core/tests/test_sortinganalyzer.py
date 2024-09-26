import pytest
from pathlib import Path
import numpy as np

import shutil

from spikeinterface.core import (
    generate_ground_truth_recording,
    create_sorting_analyzer,
    load_sorting_analyzer,
    get_available_analyzer_extensions,
    get_default_analyzer_extension_params,
)
from spikeinterface.core.sortinganalyzer import (
    register_result_extension,
    AnalyzerExtension,
    _sort_extensions_by_dependency,
)

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


@pytest.fixture(scope="module")
def dataset():
    return get_dataset()


def test_SortingAnalyzer_memory(tmp_path, dataset):
    recording, sorting = dataset
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


def test_SortingAnalyzer_binary_folder(tmp_path, dataset):
    recording, sorting = dataset

    folder = tmp_path / "test_SortingAnalyzer_binary_folder"
    if folder.exists():
        shutil.rmtree(folder)

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="binary_folder", folder=folder, sparse=False, sparsity=None
    )

    sorting_analyzer.compute(["random_spikes", "templates"])
    sorting_analyzer = load_sorting_analyzer(folder, format="auto")
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    # test select_units see https://github.com/SpikeInterface/spikeinterface/issues/3041
    # this bug requires that we have an info.json file so we calculate templates above
    select_units_sorting_analyer = sorting_analyzer.select_units(unit_ids=[1])
    assert len(select_units_sorting_analyer.unit_ids) == 1

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


def test_SortingAnalyzer_zarr(tmp_path, dataset):
    recording, sorting = dataset

    folder = tmp_path / "test_SortingAnalyzer_zarr.zarr"
    if folder.exists():
        shutil.rmtree(folder)

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="zarr", folder=folder, sparse=False, sparsity=None
    )
    sorting_analyzer.compute(["random_spikes", "templates"])
    sorting_analyzer = load_sorting_analyzer(folder, format="auto")
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    # test select_units see https://github.com/SpikeInterface/spikeinterface/issues/3041
    # this bug requires that we have an info.json file so we calculate templates above
    select_units_sorting_analyer = sorting_analyzer.select_units(unit_ids=[1])
    assert len(select_units_sorting_analyer.unit_ids) == 1
    remove_units_sorting_analyer = sorting_analyzer.remove_units(remove_unit_ids=[1])
    assert len(remove_units_sorting_analyer.unit_ids) == len(sorting_analyzer.unit_ids) - 1
    assert 1 not in remove_units_sorting_analyer.unit_ids

    folder = tmp_path / "test_SortingAnalyzer_zarr.zarr"
    if folder.exists():
        shutil.rmtree(folder)
    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="zarr", folder=folder, sparse=False, sparsity=None, return_scaled=False
    )


def test_load_without_runtime_info(tmp_path, dataset):
    import zarr

    recording, sorting = dataset

    folder = tmp_path / "test_SortingAnalyzer_run_info"

    extensions = ["random_spikes", "templates"]
    # binary_folder
    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="binary_folder", folder=folder, sparse=False, sparsity=None
    )
    sorting_analyzer.compute(extensions)
    # remove run_info.json to mimic a previous version of spikeinterface
    for ext in extensions:
        (folder / "extensions" / ext / "run_info.json").unlink()
    # should raise a warning for missing run_info
    with pytest.warns(UserWarning):
        sorting_analyzer = load_sorting_analyzer(folder, format="auto")

    # zarr
    folder = tmp_path / "test_SortingAnalyzer_run_info.zarr"
    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="zarr", folder=folder, sparse=False, sparsity=None
    )
    sorting_analyzer.compute(extensions)
    # remove run_info from attrs to mimic a previous version of spikeinterface
    root = sorting_analyzer._get_zarr_root(mode="r+")
    for ext in extensions:
        del root["extensions"][ext].attrs["run_info"]
        zarr.consolidate_metadata(root.store)
    # should raise a warning for missing run_info
    with pytest.warns(UserWarning):
        sorting_analyzer = load_sorting_analyzer(folder, format="auto")


def test_SortingAnalyzer_tmp_recording(dataset):
    recording, sorting = dataset
    recording_cached = recording.save(mode="memory")

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=False, sparsity=None)
    sorting_analyzer.set_temporary_recording(recording_cached)
    assert sorting_analyzer.has_temporary_recording()
    # check that saving as uses the original recording
    sorting_analyzer_saved = sorting_analyzer.save_as(format="memory")
    assert sorting_analyzer_saved.has_recording()
    assert not sorting_analyzer_saved.has_temporary_recording()
    assert isinstance(sorting_analyzer_saved.recording, type(recording))

    recording_sliced = recording.channel_slice(recording.channel_ids[:-1])

    # wrong channels
    with pytest.raises(ValueError):
        sorting_analyzer.set_temporary_recording(recording_sliced)


def test_SortingAnalyzer_interleaved_probegroup(dataset):
    from probeinterface import generate_linear_probe, ProbeGroup

    recording, sorting = dataset
    num_channels = recording.get_num_channels()
    probe1 = generate_linear_probe(num_elec=num_channels // 2, ypitch=20.0)
    probe2 = probe1.copy()
    probe2.move([100.0, 100.0])

    probegroup = ProbeGroup()
    probegroup.add_probe(probe1)
    probegroup.add_probe(probe2)
    probegroup.set_global_device_channel_indices(np.random.permutation(num_channels))

    recording = recording.set_probegroup(probegroup)

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    # check that locations are correct
    assert np.array_equal(recording.get_channel_locations(), sorting_analyzer.get_channel_locations())


def _check_sorting_analyzers(sorting_analyzer, original_sorting, cache_folder):

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
        assert isinstance(data["result_one"], str)
        assert isinstance(data["result_two"], np.ndarray)
        assert data["result_two"].size == original_sorting.to_spike_vector().size
        assert np.array_equal(data["result_two"], sorting_analyzer.get_extension("dummy").data["result_two"])

        assert sorting_analyzer2.return_scaled == sorting_analyzer.return_scaled

        assert sorting_analyzer2.sparsity == sorting_analyzer.sparsity

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

        # check propagation of result data and correct aligin
        assert np.array_equal(keep_unit_ids, sorting_analyzer2.unit_ids)
        data = sorting_analyzer2.get_extension("dummy").data
        assert data["result_one"] == sorting_analyzer.get_extension("dummy").data["result_one"]
        # unit 1, 3, ... should be removed
        assert np.all(~np.isin(data["result_two"], [1, 3]))

        # remove unit_ids to several format
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_remove_units_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_remove_units_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        # compute one extension to check the slice
        sorting_analyzer.compute("dummy")
        remove_unit_ids = original_sorting.unit_ids[::2]
        sorting_analyzer3 = sorting_analyzer.remove_units(remove_unit_ids=remove_unit_ids, format=format, folder=folder)

        # check propagation of result data and correct aligin
        assert np.array_equal(original_sorting.unit_ids[1::2], sorting_analyzer3.unit_ids)
        data = sorting_analyzer3.get_extension("dummy").data
        assert data["result_one"] == sorting_analyzer.get_extension("dummy").data["result_one"]
        # unit 0, 2, ... should be removed
        assert np.all(~np.isin(data["result_two"], [0, 2]))

        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_merge_soft_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_merge_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        sorting_analyzer4 = sorting_analyzer.merge_units(merge_unit_groups=[[0, 1]], format=format, folder=folder)

        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_merge_hard_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_merge_hard_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        sorting_analyzer5 = sorting_analyzer.merge_units(
            merge_unit_groups=[[0, 1]], new_unit_ids=[50], format=format, folder=folder, mode="hard"
        )

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
            # print(ext, default_params)
        else:
            try:
                default_params = get_default_analyzer_extension_params(ext)
                # print(ext, default_params)
            except:
                print(f"Failed to import {ext}")


class DummyAnalyzerExtension(AnalyzerExtension):
    extension_name = "dummy"
    depend_on = []
    need_recording = False
    use_nodepipeline = False

    def _set_params(self, param0="yep", param1=1.2, param2=[1, 2, 3.0]):
        params = dict(param0=param0, param1=param1, param2=param2)
        return params

    def _run(self, **kwargs):
        # print("dummy run")
        self.data["result_one"] = "abcd"
        # the result two has the same size of the spike vector!!
        # and represent nothing (the trick is to use unit_index for testing slice)
        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        self.data["result_two"] = spikes["unit_index"].copy()
        self.data["result_three"] = np.zeros((len(self.sorting_analyzer.unit_ids), 2))

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)
        # here the first key do not depend on unit_id
        # but the second need to be sliced!!
        new_data = dict()
        new_data["result_one"] = self.data["result_one"]
        new_data["result_two"] = self.data["result_two"][keep_spike_mask]

        keep_spike_mask = np.isin(self.sorting_analyzer.unit_ids, unit_ids)
        new_data["result_three"] = self.data["result_three"][keep_spike_mask]

        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):

        all_new_unit_ids = new_sorting_analyzer.unit_ids
        new_data = dict()
        new_data["result_one"] = self.data["result_one"]
        new_data["result_two"] = self.data["result_two"]

        arr = self.data["result_three"]
        num_dims = arr.shape[1]
        new_data["result_three"] = np.zeros((len(all_new_unit_ids), num_dims), dtype=arr.dtype)
        for unit_ind, unit_id in enumerate(all_new_unit_ids):
            if unit_id not in new_unit_ids:
                keep_unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                new_data["result_three"][unit_ind] = arr[keep_unit_index]
            else:
                id = np.flatnonzero(new_unit_ids == unit_id)[0]
                keep_unit_indices = self.sorting_analyzer.sorting.ids_to_indices(merge_unit_groups[id])
                new_data["result_three"][unit_ind] = arr[keep_unit_indices].mean(axis=0)

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


def test_extensions_sorting():

    # nothing happens if all parents are on the left of the children
    extensions_in_order = {"random_spikes": {"rs": 1}, "waveforms": {"wv": 2}}
    sorted_extensions_1 = _sort_extensions_by_dependency(extensions_in_order)
    assert list(sorted_extensions_1.keys()) == list(extensions_in_order.keys())

    extensions_out_of_order = {"waveforms": {"wv": 2}, "random_spikes": {"rs": 1}}
    sorted_extensions_2 = _sort_extensions_by_dependency(extensions_out_of_order)
    assert list(sorted_extensions_2.keys()) == list(extensions_in_order.keys())

    # doing two movements
    extensions_qm_left = {"quality_metrics": {}, "waveforms": {}, "templates": {}}
    extensions_qm_correct = {"waveforms": {}, "templates": {}, "quality_metrics": {}}
    sorted_extensions_3 = _sort_extensions_by_dependency(extensions_qm_left)
    assert list(sorted_extensions_3.keys()) == list(extensions_qm_correct.keys())

    # should move parent (waveforms) left of child (quality_metrics), and move grandparent (random_spikes) left of parent
    extensions_qm_left = {"quality_metrics": {}, "waveforms": {}, "templates": {}, "random_spikes": {}}
    extensions_qm_correct = {"random_spikes": {}, "waveforms": {}, "templates": {}, "quality_metrics": {}}
    sorted_extensions_4 = _sort_extensions_by_dependency(extensions_qm_left)
    assert list(sorted_extensions_4.keys()) == list(extensions_qm_correct.keys())


if __name__ == "__main__":
    tmp_path = Path("test_SortingAnalyzer")
    dataset = get_dataset()
    test_SortingAnalyzer_memory(tmp_path, dataset)
    test_SortingAnalyzer_binary_folder(tmp_path, dataset)
    test_SortingAnalyzer_zarr(tmp_path, dataset)
    test_SortingAnalyzer_tmp_recording(dataset)
    test_extension()
    test_SortingAnalyzer_merge_all_extensions()
    test_extension_params()
