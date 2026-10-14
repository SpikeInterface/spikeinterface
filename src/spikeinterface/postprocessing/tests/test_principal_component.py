import pytest
import numpy as np

from spikeinterface.postprocessing import ComputePrincipalComponents
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.core.tests.test_sortinganalyzer import get_dataset
from spikeinterface.core import create_sorting_analyzer


class TestPrincipalComponentsExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(mode="by_channel_local"),
            dict(mode="by_channel_global"),
            dict(mode="by_channel_global", max_fit_batch_memory="32Ki"),
            # mode concatenated cannot be tested here because it do not work with sparse=True
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputePrincipalComponents, params=params)

    def test_multi_processing(self):
        """
        Test the extension works with multiple processes.
        """
        sorting_analyzer = self._prepare_sorting_analyzer(
            format="memory", sparse=False, extension_class=ComputePrincipalComponents
        )
        sorting_analyzer.compute("principal_components", mode="by_channel_local", n_jobs=2)
        sorting_analyzer.compute(
            "principal_components", mode="by_channel_local", n_jobs=2, max_threads_per_worker=4, mp_context="spawn"
        )

    def test_mode_concatenated(self):
        """
        Replicate the "extension_function_params_list" test outside of
        AnalyzerExtensionCommonTestSuite because it does not support sparsity.

        Also, add two additional checks on the dimension and n components of the output.
        """
        sorting_analyzer = self._prepare_sorting_analyzer(
            format="memory", sparse=False, extension_class=ComputePrincipalComponents
        )

        n_components = 3
        sorting_analyzer.compute("principal_components", mode="concatenated", n_components=n_components)
        ext = sorting_analyzer.get_extension("principal_components")
        assert ext is not None
        assert len(ext.data) > 0
        pca = ext.data["pca_projection"]
        assert pca.ndim == 2
        assert pca.shape[1] == n_components

        ext_rand = sorting_analyzer.get_extension("random_spikes")
        num_rand_spikes = len(ext_rand.get_data())

        some_projections = ext.get_some_projections()
        assert some_projections[0].shape[0] == num_rand_spikes
        assert some_projections[0].shape[1] == n_components

    @pytest.mark.parametrize("sparse", [True, False])
    def test_get_projections(self, sparse):
        """
        Test the shape of output projection score matrices are
        correct when adjusting sparsity and using the
        `get_some_projections()` function. We expect them
        to hold, for each spike and each channel, the loading
        for each of the specified number of components.
        """
        sorting_analyzer = self._prepare_sorting_analyzer(
            format="memory", sparse=sparse, extension_class=ComputePrincipalComponents
        )
        num_chans = sorting_analyzer.get_num_channels()
        n_components = 2

        sorting_analyzer.compute("principal_components", mode="by_channel_global", n_components=n_components)
        ext = sorting_analyzer.get_extension("principal_components")

        # First, check the created projections have the expected number
        # of components and the expected number of channels based on sparsity.
        for unit_id in sorting_analyzer.unit_ids:
            if not sparse:
                one_proj = ext.get_projections_one_unit(unit_id, sparse=False)
                assert one_proj.shape[1] == n_components
                assert one_proj.shape[2] == num_chans
            else:
                one_proj = ext.get_projections_one_unit(unit_id, sparse=False)
                assert one_proj.shape[1] == n_components
                assert one_proj.shape[2] == num_chans

                one_proj, chan_inds = ext.get_projections_one_unit(unit_id, sparse=True)
                assert one_proj.shape[1] == n_components
                num_channels_for_unit = sorting_analyzer.sparsity.unit_id_to_channel_ids[unit_id].size
                assert one_proj.shape[2] == num_channels_for_unit
                assert one_proj.shape[2] == chan_inds.size

        # Next, check that the `get_some_projections()` function returns
        # projections with the expected shapes when selecting subjsets
        # of channel and unit IDs.
        some_unit_ids = sorting_analyzer.unit_ids[::2]
        some_channel_ids = sorting_analyzer.channel_ids[::2]

        random_spikes_ext = sorting_analyzer.get_extension("random_spikes")
        random_spikes_indices = random_spikes_ext.get_data()
        unit_ids_num_random_spikes = sum(random_spikes_ext.params["max_spikes_per_unit"] for _ in some_unit_ids)

        # this should be all spikes all channels
        some_projections, spike_unit_index = ext.get_some_projections(channel_ids=None, unit_ids=None)
        assert some_projections.shape[0] == spike_unit_index.shape[0]
        assert spike_unit_index.shape[0] == random_spikes_indices.size
        assert some_projections.shape[1] == n_components
        assert some_projections.shape[2] == num_chans

        # this should be some spikes all channels
        some_projections, spike_unit_index = ext.get_some_projections(channel_ids=None, unit_ids=some_unit_ids)
        assert some_projections.shape[0] == spike_unit_index.shape[0]
        assert spike_unit_index.shape[0] == unit_ids_num_random_spikes
        assert some_projections.shape[1] == n_components
        assert some_projections.shape[2] == num_chans
        assert 1 not in spike_unit_index

        # this should be some spikes some channels
        some_projections, spike_unit_index = ext.get_some_projections(
            channel_ids=some_channel_ids, unit_ids=some_unit_ids
        )
        assert some_projections.shape[0] == spike_unit_index.shape[0]
        assert spike_unit_index.shape[0] == unit_ids_num_random_spikes
        assert some_projections.shape[1] == n_components
        assert some_projections.shape[2] == some_channel_ids.size
        assert 1 not in spike_unit_index

        # check correctness
        channel_indices = sorting_analyzer.recording.ids_to_indices(some_channel_ids)
        for unit_id in some_unit_ids:
            unit_index = sorting_analyzer.sorting.id_to_index(unit_id)
            spike_mask = spike_unit_index == unit_index
            proj_one_unit = ext.get_projections_one_unit(unit_id, sparse=False)
            np.testing.assert_array_almost_equal(some_projections[spike_mask], proj_one_unit[:, :, channel_indices])

    @pytest.mark.parametrize("sparse", [True, False])
    def test_compute_for_all_spikes(self, sparse):
        """
        Compute the principal component scores, checking the shape
        matches the number of spikes as expected. This is re-run
        with n_jobs=2 and output projection score matrices
        checked against n_jobs=1.
        """
        sorting_analyzer = self._prepare_sorting_analyzer(
            format="memory", sparse=sparse, extension_class=ComputePrincipalComponents
        )

        num_spikes = sorting_analyzer.sorting.to_spike_vector().size

        n_components = 3
        sorting_analyzer.compute("principal_components", mode="by_channel_local", n_components=n_components)
        ext = sorting_analyzer.get_extension("principal_components")

        pc_file1 = self.cache_folder / "all_pc1.npy"
        ext.run_for_all_spikes(pc_file1, chunk_size=10000, n_jobs=1)
        all_pc1 = np.load(pc_file1)
        assert all_pc1.shape[0] == num_spikes

        pc_file2 = self.cache_folder / "all_pc2.npy"
        ext.run_for_all_spikes(pc_file2, chunk_size=10000, n_jobs=2)
        all_pc2 = np.load(pc_file2)

        np.testing.assert_almost_equal(all_pc1, all_pc2, decimal=3)

    def test_project_new(self):
        """
        `project_new` projects new (unseen) waveforms onto the PCA components.
        First compute principal components from existing waveforms. Then,
        generate a new 'spikes' vector that includes sample_index, unit_index
        and segment_index alongside some waveforms (the spike vector is required
        to generate some corresponding unit IDs for the generated waveforms following
        the API of principal_components.py).

        Then, check that the new projection scores matrix is the expected shape.
        """
        sorting_analyzer = self._prepare_sorting_analyzer(
            format="memory", sparse=False, extension_class=ComputePrincipalComponents
        )

        waveforms = sorting_analyzer.get_extension("waveforms").data["waveforms"]

        n_components = 3
        sorting_analyzer.compute("principal_components", mode="by_channel_local", n_components=n_components)
        ext_pca = sorting_analyzer.get_extension(ComputePrincipalComponents.extension_name)

        num_spike = 100
        new_spikes = sorting_analyzer.sorting.to_spike_vector()[:num_spike]
        new_waveforms = np.random.randn(num_spike, waveforms.shape[1], waveforms.shape[2])
        new_proj = ext_pca.project_new(new_spikes, new_waveforms)

        assert new_proj.shape[0] == num_spike
        assert new_proj.shape[1] == n_components
        assert new_proj.shape[2] == ext_pca.data["pca_projection"].shape[2]


def test_select_channels_sparse_pca():
    """
    Test that `_select_channels` selects the correct principal components when the analyzer
    is sparse.

    The actual code uses fancy indexing etc, so this test is designed to _not_ do this, and instead
    just loop over all units and channels to check consistency.
    """

    recording, sorting = get_dataset()
    # Make a sparse analyzer
    sorting_analyzer = create_sorting_analyzer(
        sorting, recording, format="memory", sparse=True, sparsity_kwargs={"method": "radius", "radius_um": 30}
    )
    sorting_analyzer.compute(["random_spikes", "waveforms", "principal_components"])

    # Select channels, in a non-monotonic way
    select_channel_ids = np.array(["3", "8", "7"])
    analyzer_selected = sorting_analyzer._select_channels(channel_ids=select_channel_ids)

    # Prepare the data
    original_pca = sorting_analyzer.get_extension("principal_components")
    selected_pca = analyzer_selected.get_extension("principal_components")

    for unit_id in sorting_analyzer.unit_ids:

        original_units_to_channels = sorting_analyzer.sparsity.unit_id_to_channel_ids[unit_id]
        selected_units_to_channels = analyzer_selected.sparsity.unit_id_to_channel_ids[unit_id]

        original_pca_one_unit, _ = original_pca.get_projections_one_unit(unit_id, sparse=True)
        selected_pca_one_unit, _ = selected_pca.get_projections_one_unit(unit_id, sparse=True)

        for channel_id in select_channel_ids:
            if channel_id in original_units_to_channels:

                # Now check waveforms and PCs, which are sparse
                channel_index_in_original = np.where(original_units_to_channels == channel_id)[0][0]
                original_unit_pca = original_pca_one_unit[:, :, channel_index_in_original]

                channel_index_in_selected = np.where(selected_units_to_channels == channel_id)[0][0]
                selected_unit_pca = selected_pca_one_unit[:, :, channel_index_in_selected]

                assert np.all(original_unit_pca == selected_unit_pca)


if __name__ == "__main__":
    test = TestPrincipalComponentsExtension()
    test.test_get_projections(sparse=True)


def _make_global_pca_batch_analyzer(sparse):
    from spikeinterface.core import ChannelSparsity, NumpyRecording, NumpySorting

    rng = np.random.default_rng(42)
    recording = NumpyRecording(
        [rng.normal(size=(400, 4)).astype("float32"), rng.normal(size=(200, 4)).astype("float32")], 1000.0
    )
    recording.set_dummy_probe_from_locations([[0, 0], [20, 0], [40, 0], [60, 0]])
    sorting = NumpySorting.from_unit_dict(
        [
            {10: np.arange(20, 200, 10), 30: np.array([250]), 90: np.arange(210, 350, 20)},
            {10: np.arange(20, 100, 10), 30: np.array([], dtype="int64"), 90: np.array([], dtype="int64")},
        ],
        1000.0,
    )
    sparsity = (
        ChannelSparsity(
            np.array([[1, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=bool), sorting.unit_ids, recording.channel_ids
        )
        if sparse
        else None
    )
    analyzer = create_sorting_analyzer(
        sorting, recording, sparse=False, sparsity=sparsity, main_channel_indices=np.array([0, 1, 2])
    )
    analyzer.compute(
        {"random_spikes": {"method": "all"}, "waveforms": {"ms_before": 3, "ms_after": 4}}, n_jobs=1, progress_bar=False
    )
    return analyzer


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("storage", ["numpy", "zarr"])
def test_global_pca_fit_batches(tmp_path, monkeypatch, sparse, storage):
    from sklearn.decomposition import IncrementalPCA
    import spikeinterface.postprocessing.principal_component as pc_module

    analyzer = _make_global_pca_batch_analyzer(sparse)
    waveforms_ext = analyzer.get_extension("waveforms")
    waveforms = waveforms_ext.data["waveforms"]
    if storage == "zarr":
        import zarr

        stored = zarr.open_array(
            str(tmp_path / "waveforms.zarr"),
            mode="w",
            shape=waveforms.shape,
            dtype=waveforms.dtype,
            chunks=(4, *waveforms.shape[1:]),
        )
        stored[:] = waveforms
        waveforms_ext.data["waveforms"] = stored
    ext = ComputePrincipalComponents(analyzer)
    ext.set_params(n_components=3, mode="by_channel_global")
    expected_inputs = []
    original = IncrementalPCA(n_components=3, whiten=True)
    for uid in analyzer.unit_ids:
        wfs, _, _ = ext._get_sparse_waveforms(uid)
        flat = wfs.transpose(0, 2, 1).reshape(-1, wfs.shape[1])
        if len(flat) >= 3:
            expected_inputs.append(flat)
            original.partial_fit(flat)
    # Units below the default budget keep exactly the previous fitting updates.
    unchanged = ext._fit_by_channel_global(progress_bar=False)
    np.testing.assert_array_equal(unchanged.components_, original.components_)
    np.testing.assert_array_equal(unchanged.singular_values_, original.singular_values_)

    batch_inputs, read_sizes = [], []
    real_fit, real_slice = IncrementalPCA.partial_fit, pc_module.slice_rows

    def partial_fit(self, data, **kwargs):
        batch_inputs.append(data.copy())
        assert data.nbytes <= 512
        assert len(data) >= 3
        return real_fit(self, data, **kwargs)

    def slice_rows(array, indices):
        selected = real_slice(array, indices)
        read_sizes.append(selected.nbytes)
        assert selected.nbytes <= 512, "A whole unit was read before batching"
        return selected

    monkeypatch.setattr(IncrementalPCA, "partial_fit", partial_fit)
    monkeypatch.setattr(pc_module, "slice_rows", slice_rows)
    ext.params["max_fit_batch_memory"] = "0.5Ki"
    fitted = ext._fit_by_channel_global(progress_bar=False)
    expected = np.concatenate(expected_inputs)
    np.testing.assert_array_equal(np.concatenate(batch_inputs), expected)
    assert len(batch_inputs) > len(expected_inputs)
    assert fitted.n_samples_seen_ == len(expected)
    np.testing.assert_allclose(fitted.mean_, np.mean(expected, axis=0), atol=1e-6)
    assert read_sizes


def test_global_pca_fit_batch_memory_validation():
    analyzer = _make_global_pca_batch_analyzer(sparse=True)
    ext = ComputePrincipalComponents(analyzer)
    for value in ("0Ki", "-1Ki"):
        with pytest.raises(ValueError, match="max_fit_batch_memory must be positive"):
            ext.set_params(mode="by_channel_global", max_fit_batch_memory=value)
    ext.set_params(n_components=3, mode="by_channel_global", max_fit_batch_memory="0.001Ki")
    with pytest.raises(ValueError, match="too small"):
        ext._fit_by_channel_global(progress_bar=False)
    # Seven one-channel waveforms cannot be partitioned into batches of at most
    # three waveforms with at least three rows each; report an actionable error.
    ext.set_params(n_components=3, mode="by_channel_global", max_fit_batch_memory="0.25Ki")
    with pytest.raises(ValueError, match="Increase max_fit_batch_memory"):
        ext._fit_by_channel_global(progress_bar=False)
