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
    get_default_zarr_compressor,
)
from spikeinterface.core.sortinganalyzer import (
    register_result_extension,
    AnalyzerExtension,
    _sort_extensions_by_dependency,
)
from spikeinterface.core.analyzer_extension_core import BaseSpikeVectorExtension

# to test basespikevectorextension with node pipeline
from spikeinterface.core.node_pipeline import SpikeRetriever
from spikeinterface.core.tests.test_node_pipeline import AmplitudeExtractionNode


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

    # TODO: the tests or the sorting analyzer make assumptions about the ids being integers
    # So keeping this the way it was
    integer_channel_ids = [int(id) for id in recording.get_channel_ids()]
    integer_unit_ids = [int(id) for id in sorting.get_unit_ids()]

    recording = recording.rename_channels(new_channel_ids=integer_channel_ids)
    sorting = sorting.rename_units(new_unit_ids=integer_unit_ids)
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
        sorting, recording, format="memory", sparse=False, return_in_uV=True, sparsity=None
    )
    assert sorting_analyzer.return_in_uV
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="memory", sparse=False, return_in_uV=False, sparsity=None
    )
    assert not sorting_analyzer.return_in_uV

    # test set_sorting_property
    sorting_analyzer.set_sorting_property(key="quality", values=["good"] * len(sorting_analyzer.unit_ids))
    sorting_analyzer.set_sorting_property(key="number", values=np.arange(len(sorting_analyzer.unit_ids)))
    assert "quality" in sorting_analyzer.sorting.get_property_keys()
    assert "number" in sorting_analyzer.sorting.get_property_keys()


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
        return_in_uV=False,
    )
    assert not sorting_analyzer.return_in_uV
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    # test set_sorting_property
    sorting_analyzer.set_sorting_property(key="quality", values=["good"] * len(sorting_analyzer.unit_ids))
    sorting_analyzer.set_sorting_property(key="number", values=np.arange(len(sorting_analyzer.unit_ids)))
    assert "quality" in sorting_analyzer.sorting.get_property_keys()
    assert "number" in sorting_analyzer.sorting.get_property_keys()
    sorting_analyzer_reloded = load_sorting_analyzer(folder, format="auto")
    assert "quality" in sorting_analyzer_reloded.sorting.get_property_keys()
    assert "number" in sorting_analyzer.sorting.get_property_keys()


def test_SortingAnalyzer_zarr(tmp_path, dataset):
    recording, sorting = dataset

    folder = tmp_path / "test_SortingAnalyzer_zarr.zarr"

    default_compressor = get_default_zarr_compressor()
    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="zarr", folder=folder, sparse=False, sparsity=None, overwrite=True
    )
    sorting_analyzer.compute(["random_spikes", "templates"])
    sorting_analyzer = load_sorting_analyzer(folder, format="auto")
    _check_sorting_analyzers(sorting_analyzer, sorting, cache_folder=tmp_path)

    # check that compression is applied
    assert (
        sorting_analyzer._get_zarr_root()["extensions"]["random_spikes"]["random_spikes_indices"].compressor.codec_id
        == default_compressor.codec_id
    )
    assert (
        sorting_analyzer._get_zarr_root()["extensions"]["templates"]["average"].compressor.codec_id
        == default_compressor.codec_id
    )

    # test select_units see https://github.com/SpikeInterface/spikeinterface/issues/3041
    # this bug requires that we have an info.json file so we calculate templates above
    select_units_sorting_analyer = sorting_analyzer.select_units(unit_ids=[1])
    assert len(select_units_sorting_analyer.unit_ids) == 1
    remove_units_sorting_analyer = sorting_analyzer.remove_units(remove_unit_ids=[1])
    assert len(remove_units_sorting_analyer.unit_ids) == len(sorting_analyzer.unit_ids) - 1
    assert 1 not in remove_units_sorting_analyer.unit_ids

    # test no compression
    sorting_analyzer_no_compression = create_sorting_analyzer(
        sorting,
        recording,
        format="zarr",
        folder=folder,
        sparse=False,
        sparsity=None,
        return_in_uV=False,
        overwrite=True,
        backend_options={"saving_options": {"compressor": None}},
    )
    print(sorting_analyzer_no_compression._backend_options)
    sorting_analyzer_no_compression.compute(["random_spikes", "templates"])
    assert (
        sorting_analyzer_no_compression._get_zarr_root()["extensions"]["random_spikes"][
            "random_spikes_indices"
        ].compressor
        is None
    )
    assert sorting_analyzer_no_compression._get_zarr_root()["extensions"]["templates"]["average"].compressor is None

    # test a different compressor
    from numcodecs import LZMA

    lzma_compressor = LZMA()
    folder = tmp_path / "test_SortingAnalyzer_zarr_lzma.zarr"
    sorting_analyzer_lzma = sorting_analyzer_no_compression.save_as(
        format="zarr", folder=folder, backend_options={"saving_options": {"compressor": lzma_compressor}}
    )
    assert (
        sorting_analyzer_lzma._get_zarr_root()["extensions"]["random_spikes"][
            "random_spikes_indices"
        ].compressor.codec_id
        == LZMA.codec_id
    )
    assert (
        sorting_analyzer_lzma._get_zarr_root()["extensions"]["templates"]["average"].compressor.codec_id
        == LZMA.codec_id
    )

    # test set_sorting_property
    sorting_analyzer.set_sorting_property(key="quality", values=["good"] * len(sorting_analyzer.unit_ids))
    sorting_analyzer.set_sorting_property(key="number", values=np.arange(len(sorting_analyzer.unit_ids)))
    assert "quality" in sorting_analyzer.sorting.get_property_keys()
    assert "number" in sorting_analyzer.sorting.get_property_keys()
    sorting_analyzer_reloded = load_sorting_analyzer(sorting_analyzer.folder, format="auto")
    assert "quality" in sorting_analyzer_reloded.sorting.get_property_keys()
    assert "number" in sorting_analyzer.sorting.get_property_keys()


def test_create_by_dict():
    """
    Generates a recording and sorting which are split into dicts and fed to create_sorting_analyzer.
    Interally, this aggregates the dicts of recordings and sortings. This test checks that the
    unit structure is maintained from the dicts to the analyzer. Then checks that the function
    fails if the dict keys are different for the recordings and the sortings.
    """

    rec, sort = generate_ground_truth_recording(num_channels=6)

    rec.set_property(key="group", values=[1, 2, 1, 1, 2, 2])
    sort.set_property(key="group", values=[2, 2, 2, 1, 2, 2, 2, 1, 2, 1])

    unit_ids = sort.unit_ids
    split_sort = sort.split_by("group")
    split_rec = rec.split_by("group")
    analyzer = create_sorting_analyzer(split_sort, split_rec)
    analyzer_unit_ids = analyzer.unit_ids

    assert set(analyzer.unit_ids) == set(sort.unit_ids)
    assert np.all(analyzer_unit_ids[analyzer.get_sorting_property("group") == 1] == split_sort[1].unit_ids)
    assert np.all(analyzer_unit_ids[analyzer.get_sorting_property("group") == 2] == split_sort[2].unit_ids)

    assert np.all(sort.get_unit_spike_train(unit_id="5") == analyzer.sorting.get_unit_spike_train(unit_id="5"))

    # make a dict of sortings with keys which don't match the recordings keys
    split_sort_bad_keys = {
        bad_key: sort.select_units(unit_ids=unit_ids[sort.get_property("group") == key])
        for bad_key, key in zip([3, 4], [1, 2])
    }

    with pytest.raises(ValueError):
        analyzer = create_sorting_analyzer(split_sort_bad_keys, rec.split_by("group"))

    # make a dict of sortings, in a different order than the recording. This should
    # still work
    split_sort_different_order = {
        2: sort.select_units(unit_ids=unit_ids[sort.get_property("group") == 2]),
        1: sort.select_units(unit_ids=unit_ids[sort.get_property("group") == 1]),
    }
    combined_analyzer = create_sorting_analyzer(split_sort_different_order, rec.split_by("group"))
    assert np.all(sort.get_unit_spike_train(unit_id="5") == combined_analyzer.sorting.get_unit_spike_train(unit_id="5"))


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

    recording_sliced = recording.select_channels(recording.channel_ids[:-1])

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

        assert sorting_analyzer2.return_in_uV == sorting_analyzer.return_in_uV

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

        # test merges
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_merge_soft_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_merge_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        sorting_analyzer4, new_unit_ids = sorting_analyzer.merge_units(
            merge_unit_groups=[[0, 1]], format=format, folder=folder, return_new_unit_ids=True
        )
        assert 0 not in sorting_analyzer4.unit_ids
        assert 1 not in sorting_analyzer4.unit_ids
        assert len(sorting_analyzer4.unit_ids) == len(sorting_analyzer.unit_ids) - 1
        is_merged_values = sorting_analyzer4.sorting.get_property("is_merged")
        assert is_merged_values[sorting_analyzer4.sorting.ids_to_indices(new_unit_ids)][0]

        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_merge_hard_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_merge_hard_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        sorting_analyzer5, new_unit_ids = sorting_analyzer.merge_units(
            merge_unit_groups=[[0, 1]],
            new_unit_ids=[50],
            format=format,
            folder=folder,
            merging_mode="hard",
            return_new_unit_ids=True,
        )
        assert 0 not in sorting_analyzer5.unit_ids
        assert 1 not in sorting_analyzer5.unit_ids
        assert len(sorting_analyzer5.unit_ids) == len(sorting_analyzer.unit_ids) - 1
        assert 50 in sorting_analyzer5.unit_ids
        is_merged_values = sorting_analyzer5.sorting.get_property("is_merged")
        assert is_merged_values[sorting_analyzer5.sorting.id_to_index(50)]

        # test splitting
        if format != "memory":
            if format == "zarr":
                folder = cache_folder / f"test_SortingAnalyzer_split_with_{format}.zarr"
            else:
                folder = cache_folder / f"test_SortingAnalyzer_split_with_{format}"
            if folder.exists():
                shutil.rmtree(folder)
        else:
            folder = None
        split_units = {}
        num_spikes = sorting_analyzer.sorting.count_num_spikes_per_unit()
        units_to_split = sorting_analyzer.unit_ids[:2]
        for unit in units_to_split:
            for unit in units_to_split:
                split_units[unit] = [
                    np.arange(num_spikes[unit] // 2),
                    np.arange(num_spikes[unit] // 2, num_spikes[unit]),
                ]

        sorting_analyzer6, split_new_unit_ids = sorting_analyzer.split_units(
            split_units=split_units, format=format, folder=folder, return_new_unit_ids=True
        )
        for unit_to_split in units_to_split:
            assert unit_to_split not in sorting_analyzer6.unit_ids
        assert len(sorting_analyzer6.unit_ids) == len(sorting_analyzer.unit_ids) + 2
        is_split_values = sorting_analyzer6.sorting.get_property("is_split")
        for new_unit_ids in split_new_unit_ids:
            assert all(is_split_values[sorting_analyzer6.sorting.ids_to_indices(new_unit_ids)])

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

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        new_data = dict()
        new_data["result_one"] = self.data["result_one"]
        spikes = new_sorting_analyzer.sorting.to_spike_vector()
        new_data["result_two"] = spikes["unit_index"].copy()
        new_data["result_three"] = np.zeros((len(new_sorting_analyzer.unit_ids), 2))
        return new_data

    def _get_data(self):
        return self.data["result_one"]


compute_dummy = DummyAnalyzerExtension.function_factory()


class DummyPipelineAnalyzerExtension(BaseSpikeVectorExtension):
    extension_name = "dummy_pipeline"
    depend_on = ["templates"]
    need_recording = True
    use_nodepipeline = True
    nodepipeline_variables = ["amp"]

    @classmethod
    def get_required_dependencies(cls, **params):
        param0 = params.get("param0", 5.5)
        if param0 > 10:
            return ["dummy"]
        else:
            return []

    def _set_params(self, param0=5.5):
        params = dict(param0=param0)
        return params

    def _get_pipeline_nodes(self):
        from spikeinterface.core.template_tools import get_template_extremum_channel

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting

        extremum_channel_inds = get_template_extremum_channel(self.sorting_analyzer, outputs="index")
        spike_retriever_node = SpikeRetriever(
            sorting, recording, channel_from_template=True, extremum_channel_inds=extremum_channel_inds
        )
        spike_amplitudes_node = AmplitudeExtractionNode(
            recording,
            parents=[spike_retriever_node],
            return_output=True,
            param0=self.params["param0"],
        )
        nodes = [spike_retriever_node, spike_amplitudes_node]
        return nodes


class DummyAnalyzerExtension2(AnalyzerExtension):
    extension_name = "dummy"


def test_extension():
    register_result_extension(DummyAnalyzerExtension)
    # can be register twice without error
    register_result_extension(DummyAnalyzerExtension)

    # other extension with same name should trigger an error
    with pytest.raises(AssertionError):
        register_result_extension(DummyAnalyzerExtension2)


def test_excess_spikes(dataset):
    """
    If there are spikes that occur after the recording end time,
    `create_sorting_analyzer` should cut them off and warn the user.
    """
    recording, sorting = dataset
    with pytest.warns(UserWarning):
        create_sorting_analyzer(sorting=sorting, recording=recording.time_slice(0, 1))


def test_extensions_sorting():

    # nothing happens if all parents are on the left of the children
    extensions_in_order = {"random_spikes": {"rs": 1}, "waveforms": {"wv": 2}}
    sorted_extensions_1 = _sort_extensions_by_dependency(extensions_in_order)
    assert list(sorted_extensions_1.keys()) == list(extensions_in_order.keys())

    extensions_out_of_order = {"waveforms": {"wv": 2}, "random_spikes": {"rs": 1}}
    sorted_extensions_2 = _sort_extensions_by_dependency(extensions_out_of_order)
    assert list(sorted_extensions_2.keys()) == list(extensions_in_order.keys())

    # doing two movements
    extensions_qm_left = {"template_metrics": {}, "waveforms": {}, "templates": {}}
    extensions_qm_correct = {"waveforms": {}, "templates": {}, "template_metrics": {}}
    sorted_extensions_3 = _sort_extensions_by_dependency(extensions_qm_left)
    assert list(sorted_extensions_3.keys()) == list(extensions_qm_correct.keys())

    # should move parent (waveforms) left of child (template_metrics), and move grandparent (random_spikes) left of parent
    extensions_qm_left = {"template_metrics": {}, "waveforms": {}, "templates": {}, "random_spikes": {}}
    extensions_qm_correct = {"random_spikes": {}, "waveforms": {}, "templates": {}, "template_metrics": {}}
    sorted_extensions_4 = _sort_extensions_by_dependency(extensions_qm_left)
    assert list(sorted_extensions_4.keys()) == list(extensions_qm_correct.keys())


def test_runtime_dependencies(dataset):
    recording, sorting = dataset
    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=False, sparsity=None)

    # param0 <=10 : no dependency
    deps = DummyPipelineAnalyzerExtension.get_required_dependencies(param0=5)
    assert deps == []

    # param0 >10 : depend on dummy
    deps = DummyPipelineAnalyzerExtension.get_required_dependencies(param0=15)
    assert deps == ["dummy"]

    register_result_extension(DummyPipelineAnalyzerExtension)
    register_result_extension(DummyAnalyzerExtension)

    sorting_analyzer.compute(["random_spikes", "templates"])
    # no dependency
    sorting_analyzer.compute("dummy_pipeline", param0=5)

    # raise if dependency not computed
    with pytest.raises(AssertionError):
        sorting_analyzer.compute("dummy_pipeline", param0=15)

    # run fine if dependency computed
    sorting_analyzer.compute(["dummy", "dummy_pipeline"], extension_params=dict(dummy_pipeline=dict(param0=11)))

    # check deletion dependency: since now dummy_pipeline depends on dummy,
    # recomputing dummy also deletes dummy_pipeline
    sorting_analyzer.compute("dummy")
    assert not sorting_analyzer.has_extension("dummy_pipeline")


if __name__ == "__main__":
    tmp_path = Path("test_SortingAnalyzer")
    dataset = get_dataset()
    test_SortingAnalyzer_memory(tmp_path, dataset)
    test_SortingAnalyzer_binary_folder(tmp_path, dataset)
    test_SortingAnalyzer_zarr(tmp_path, dataset)
    test_SortingAnalyzer_tmp_recording(dataset)
    test_extension()
    test_extension_params()
    test_runtime_dependencies()
