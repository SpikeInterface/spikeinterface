import pytest
import numpy as np

from spikeinterface.postprocessing import ComputePrincipalComponents
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite


class TestPrincipalComponentsExtension(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(mode="by_channel_local"),
            dict(mode="by_channel_global"),
            # mode concatenated cannot be tested here because it do not work with sparse=True
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputePrincipalComponents, params=params)

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

        random_spikes_indices = sorting_analyzer.get_extension("random_spikes").get_data()
        all_num_spikes = sorting_analyzer.sorting.get_total_num_spikes()
        unit_ids_num_spikes = np.sum(all_num_spikes[unit_id] for unit_id in some_unit_ids)

        # this should be all spikes all channels
        some_projections, spike_unit_index = ext.get_some_projections(channel_ids=None, unit_ids=None)
        assert some_projections.shape[0] == spike_unit_index.shape[0]
        assert spike_unit_index.shape[0] == random_spikes_indices.size
        assert some_projections.shape[1] == n_components
        assert some_projections.shape[2] == num_chans

        # this should be some spikes all channels
        some_projections, spike_unit_index = ext.get_some_projections(channel_ids=None, unit_ids=some_unit_ids)
        assert some_projections.shape[0] == spike_unit_index.shape[0]
        assert spike_unit_index.shape[0] == unit_ids_num_spikes
        assert some_projections.shape[1] == n_components
        assert some_projections.shape[2] == num_chans
        assert 1 not in spike_unit_index

        # this should be some spikes some channels
        some_projections, spike_unit_index = ext.get_some_projections(
            channel_ids=some_channel_ids, unit_ids=some_unit_ids
        )
        assert some_projections.shape[0] == spike_unit_index.shape[0]
        assert spike_unit_index.shape[0] == unit_ids_num_spikes
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


if __name__ == "__main__":
    test = TestPrincipalComponentsExtension()
    test.test_get_projections(sparse=True)
