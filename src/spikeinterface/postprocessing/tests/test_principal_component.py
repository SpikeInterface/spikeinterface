import unittest
import pytest
from pathlib import Path

import numpy as np

from spikeinterface.postprocessing import ComputePrincipalComponents, compute_principal_components
from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite, cache_folder


DEBUG = False


class PrincipalComponentsExtensionTest(AnalyzerExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputePrincipalComponents
    extension_function_params_list = [
        dict(mode="by_channel_local"),
        dict(mode="by_channel_global"),
        # mode concatenated cannot be tested here because it do not work with sparse=True
    ]

    def test_mode_concatenated(self):
        # this is tested outside "extension_function_params_list" because it do not support sparsity!

        sorting_analyzer = self._prepare_sorting_analyzer(format="memory", sparse=False)

        n_components = 3
        sorting_analyzer.compute("principal_components", mode="concatenated", n_components=n_components)
        ext = sorting_analyzer.get_extension("principal_components")
        assert ext is not None
        assert len(ext.data) > 0
        pca = ext.data["pca_projection"]
        assert pca.ndim == 2
        assert pca.shape[1] == n_components

    def test_get_projections(self):

        for sparse in (False, True):

            sorting_analyzer = self._prepare_sorting_analyzer(format="memory", sparse=sparse)
            num_chans = sorting_analyzer.get_num_channels()
            n_components = 2

            sorting_analyzer.compute("principal_components", mode="by_channel_global", n_components=n_components)
            ext = sorting_analyzer.get_extension("principal_components")

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
                    assert one_proj.shape[2] < num_chans
                    assert one_proj.shape[2] == chan_inds.size

            some_unit_ids = sorting_analyzer.unit_ids[::2]
            some_channel_ids = sorting_analyzer.channel_ids[::2]

            random_spikes_indices = sorting_analyzer.get_extension("random_spikes").get_data()

            # this should be all spikes all channels
            some_projections, spike_unit_index = ext.get_some_projections(channel_ids=None, unit_ids=None)
            assert some_projections.shape[0] == spike_unit_index.shape[0]
            assert spike_unit_index.shape[0] == random_spikes_indices.size
            assert some_projections.shape[1] == n_components
            assert some_projections.shape[2] == num_chans

            # this should be some spikes all channels
            some_projections, spike_unit_index = ext.get_some_projections(channel_ids=None, unit_ids=some_unit_ids)
            assert some_projections.shape[0] == spike_unit_index.shape[0]
            assert spike_unit_index.shape[0] < random_spikes_indices.size
            assert some_projections.shape[1] == n_components
            assert some_projections.shape[2] == num_chans
            assert 1 not in spike_unit_index

            # this should be some spikes some channels
            some_projections, spike_unit_index = ext.get_some_projections(
                channel_ids=some_channel_ids, unit_ids=some_unit_ids
            )
            assert some_projections.shape[0] == spike_unit_index.shape[0]
            assert spike_unit_index.shape[0] < random_spikes_indices.size
            assert some_projections.shape[1] == n_components
            assert some_projections.shape[2] == some_channel_ids.size
            assert 1 not in spike_unit_index

    def test_compute_for_all_spikes(self):

        for sparse in (True, False):
            sorting_analyzer = self._prepare_sorting_analyzer(format="memory", sparse=sparse)

            num_spikes = sorting_analyzer.sorting.to_spike_vector().size

            n_components = 3
            sorting_analyzer.compute("principal_components", mode="by_channel_local", n_components=n_components)
            ext = sorting_analyzer.get_extension("principal_components")

            pc_file1 = cache_folder / "all_pc1.npy"
            ext.run_for_all_spikes(pc_file1, chunk_size=10000, n_jobs=1)
            all_pc1 = np.load(pc_file1)
            assert all_pc1.shape[0] == num_spikes

            pc_file2 = cache_folder / "all_pc2.npy"
            ext.run_for_all_spikes(pc_file2, chunk_size=10000, n_jobs=2)
            all_pc2 = np.load(pc_file2)

            assert np.array_equal(all_pc1, all_pc2)

    def test_project_new(self):
        from sklearn.decomposition import IncrementalPCA

        sorting_analyzer = self._prepare_sorting_analyzer(format="memory", sparse=False)

        waveforms = sorting_analyzer.get_extension("waveforms").data["waveforms"]

        n_components = 3
        sorting_analyzer.compute("principal_components", mode="by_channel_local", n_components=n_components)
        ext_pca = sorting_analyzer.get_extension(self.extension_name)

        num_spike = 100
        new_spikes = sorting_analyzer.sorting.to_spike_vector()[:num_spike]
        new_waveforms = np.random.randn(num_spike, waveforms.shape[1], waveforms.shape[2])
        new_proj = ext_pca.project_new(new_spikes, new_waveforms)

        assert new_proj.shape[0] == num_spike
        assert new_proj.shape[1] == n_components
        assert new_proj.shape[2] == ext_pca.data["pca_projection"].shape[2]


if __name__ == "__main__":
    test = PrincipalComponentsExtensionTest()
    test.setUpClass()
    test.test_extension()
    test.test_mode_concatenated()
    test.test_get_projections()
    test.test_compute_for_all_spikes()
    test.test_project_new()

    # ext = test.sorting_analyzers["sparseTrue_memory"].get_extension("principal_components")
    # pca = ext.data["pca_projection"]
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(pca[:, 0, 0], pca[:, 0, 1])
    # plt.show()
