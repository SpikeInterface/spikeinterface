import unittest
import pytest
from pathlib import Path

import numpy as np

from spikeinterface.postprocessing import ComputePrincipalComponents, compute_principal_components
from spikeinterface.postprocessing.tests.common_extension_tests import ResultExtensionCommonTestSuite




# from spikeinterface import compute_sparsity
# from spikeinterface.postprocessing import WaveformPrincipalComponent, compute_principal_components
# from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite

# if hasattr(pytest, "global_test_folder"):
#     cache_folder = pytest.global_test_folder / "postprocessing"
# else:
#     cache_folder = Path("cache_folder") / "postprocessing"


DEBUG = False


class PrincipalComponentsExtensionTest(ResultExtensionCommonTestSuite, unittest.TestCase):
    extension_class = ComputePrincipalComponents
    extension_function_kwargs_list = [
        dict(mode="by_channel_local"),
        dict(mode="by_channel_global"),
        # mode concatenated cannot be tested here because it do not work with sparse=True
    ]

    # TODO : put back theses tests

    def test_mode_concatenated(self):
        # this is tested outside "extension_function_kwargs_list" because it do not support sparsity!

        sorting_result = self._prepare_sorting_result(format="memory", sparse=False)

        n_components = 3
        sorting_result.compute("principal_components", mode="concatenated", n_components=n_components)
        ext = sorting_result.get_extension(self.extension_name)
        assert ext is not None
        assert len(ext.data) > 0
        pca = ext.data["pca_projection"]
        assert pca.ndim == 2
        assert pca.shape[1] == n_components

    # def test_compute_for_all_spikes(self):
    #     sorting_result = self._prepare_sorting_result(format="memory", sparse=False)

    #     n_components = 3
    #     sorting_result.compute("principal_components", mode="by_channel_local", n_components=n_components)
    #     ext = sorting_result.get_extension(self.extension_name)
    #     ext.run_for_all_spikes()

    #     pc_file1 = pc.extension_folder / "all_pc1.npy"
    #     pc.run_for_all_spikes(pc_file1, chunk_size=10000, n_jobs=1)
    #     all_pc1 = np.load(pc_file1)

    #     pc_file2 = pc.extension_folder / "all_pc2.npy"
    #     pc.run_for_all_spikes(pc_file2, chunk_size=10000, n_jobs=2)
    #     all_pc2 = np.load(pc_file2)

    #     assert np.array_equal(all_pc1, all_pc2)

    #     # test with sparsity
    #     sparsity = compute_sparsity(we, method="radius", radius_um=50)
    #     we_copy = we.save(folder=cache_folder / "we_copy")
    #     pc_sparse = self.extension_class.get_extension_function()(we_copy, sparsity=sparsity, load_if_exists=False)
    #     pc_file_sparse = pc.extension_folder / "all_pc_sparse.npy"
    #     pc_sparse.run_for_all_spikes(pc_file_sparse, chunk_size=10000, n_jobs=1)
    #     all_pc_sparse = np.load(pc_file_sparse)
    #     all_spikes_seg0 = we_copy.sorting.to_spike_vector(concatenated=False)[0]
    #     for unit_index, unit_id in enumerate(we.unit_ids):
    #         sparse_channel_ids = sparsity.unit_id_to_channel_ids[unit_id]
    #         pc_unit = all_pc_sparse[all_spikes_seg0["unit_index"] == unit_index]
    #         assert np.allclose(pc_unit[:, :, len(sparse_channel_ids) :], 0)


    def test_project_new(self):
        from sklearn.decomposition import IncrementalPCA

        sorting_result = self._prepare_sorting_result(format="memory", sparse=False)

        waveforms = sorting_result.get_extension("waveforms").data["waveforms"]

        n_components = 3
        sorting_result.compute("principal_components", mode="by_channel_local", n_components=n_components)
        ext_pca = sorting_result.get_extension(self.extension_name)


        num_spike = 100
        new_spikes = sorting_result.sorting.to_spike_vector()[:num_spike]
        new_waveforms = np.random.randn(num_spike, waveforms.shape[1], waveforms.shape[2])
        new_proj = ext_pca.project_new(new_spikes, new_waveforms)
        
        assert new_proj.shape[0] == num_spike
        assert new_proj.shape[1] == n_components
        assert new_proj.shape[2] == ext_pca.data["pca_projection"].shape[2]


if __name__ == "__main__":
    test = PrincipalComponentsExtensionTest()
    test.setUp()
    test.test_extension()
    test.test_mode_concatenated()
    # test.test_compute_for_all_spikes()
    test.test_project_new()


    # ext = test.sorting_results["sparseTrue_memory"].get_extension("principal_components")
    # pca = ext.data["pca_projection"]
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(pca[:, 0, 0], pca[:, 0, 1])
    # plt.show()
