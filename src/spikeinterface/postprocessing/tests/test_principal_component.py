import unittest
import pytest
from pathlib import Path

import numpy as np

from spikeinterface import compute_sparsity
from spikeinterface.postprocessing import WaveformPrincipalComponent, compute_principal_components
from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


DEBUG = False

class PrincipalComponentsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = WaveformPrincipalComponent
    extension_data_names = ["pca_0", "pca_1"]
    extension_function_kwargs_list = [
        dict(mode='by_channel_local'),
        dict(mode='by_channel_local', n_jobs=2),
        dict(mode='by_channel_global'),
        dict(mode='concatenated'),
    ]

    def test_shapes(self):
        nchan1 = self.we1.recording.get_num_channels()
        for mode in ('by_channel_local', 'by_channel_global'):
            _ = self.extension_class.get_extension_function()(
                self.we1, mode=mode, n_components=5)
            pc = self.we1.load_extension(self.extension_class.extension_name)
            for unit_id in self.we1.sorting.unit_ids:
                proj = pc.get_projections(unit_id)
                assert proj.shape[1:] == (5, nchan1)
        for mode in ('concatenated',):
            _ = self.extension_class.get_extension_function()(
                self.we2, mode=mode, n_components=3)
            pc = self.we2.load_extension(self.extension_class.extension_name)
            for unit_id in self.we2.sorting.unit_ids:
                proj = pc.get_projections(unit_id)
                assert proj.shape[1] == 3

    def test_compute_for_all_spikes(self):
        we = self.we1
        pc = self.extension_class.get_extension_function()(we, load_if_exists=True)
        print(pc)

        pc_file1 = pc.extension_folder / 'all_pc1.npy'
        pc.run_for_all_spikes(pc_file1, chunk_size=10000, n_jobs=1)
        all_pc1 = np.load(pc_file1)

        pc_file2 = pc.extension_folder / 'all_pc2.npy'
        pc.run_for_all_spikes(pc_file2, chunk_size=10000, n_jobs=2)
        all_pc2 = np.load(pc_file2)

        assert np.array_equal(all_pc1, all_pc2)

        # test with sparsity
        sparsity = compute_sparsity(we, method="radius", radius_um=50)
        we_copy = we.save(folder=cache_folder / "we_copy")
        pc_sparse = self.extension_class.get_extension_function()(we_copy, sparsity=sparsity, load_if_exists=False)
        pc_file_sparse = pc.extension_folder / 'all_pc_sparse.npy'
        pc_sparse.run_for_all_spikes(pc_file_sparse, chunk_size=10000, n_jobs=1)
        all_pc_sparse = np.load(pc_file_sparse)
        all_spikes = we_copy.sorting.get_all_spike_trains(outputs='unit_id')
        _, spike_labels = all_spikes[0]
        for unit_id, sparse_channel_ids in sparsity.unit_id_to_channel_ids.items():
            # check dimensions
            pc_unit = all_pc_sparse[spike_labels == unit_id]
            assert np.allclose(pc_unit[:, :, len(sparse_channel_ids):], 0)

    def test_sparse(self):
        we = self.we2
        unit_ids = we.unit_ids
        num_channels = we.get_num_channels()
        pc = self.extension_class(we)
        
        sparsity_radius = compute_sparsity(we, method="radius", radius_um=50)
        sparsity_best = compute_sparsity(we, method="best_channels", num_channels=2)
        sparsities = [sparsity_radius, sparsity_best]
        print(sparsities)

        for mode in ('by_channel_local', 'by_channel_global'):
            for sparsity in sparsities:
                pc.set_params(n_components=5, mode=mode, sparsity=sparsity)
                pc.run()
                for i, unit_id in enumerate(unit_ids):
                    proj = pc.get_projections(unit_id)
                    assert proj.shape[1:] == (5, 4)

                # test project_new
                unit_id = 3
                new_wfs = we.get_waveforms(unit_id)
                new_proj = pc.project_new(new_wfs, unit_id=unit_id)
                assert new_proj.shape == (new_wfs.shape[0], 5, 4)

                if DEBUG:
                    import matplotlib.pyplot as plt
                    plt.ion()
                    cmap = plt.get_cmap('jet', len(unit_ids))
                    fig, axs = plt.subplots(nrows=len(unit_ids), ncols=num_channels)
                    for i, unit_id in enumerate(unit_ids):
                        comp = pc.get_projections(unit_id)
                        print(comp.shape)
                        for chan_ind in range(num_channels):
                            ax = axs[i, chan_ind]
                            ax.scatter(comp[:, 0, chan_ind], comp[:, 1, chan_ind], color=cmap(i))
                            ax.set_title(f"{mode}-{sparsity.unit_id_to_channel_ids[unit_id]}")
                            if i == 0:
                                ax.set_xlabel(f"Ch{chan_ind}")
                    plt.show()

        for mode in ('concatenated',):
            # concatenated is only compatible with "best"
            pc.set_params(n_components=5, mode=mode, sparsity=sparsity_best)
            print(pc)
            pc.run()
            for i, unit_id in enumerate(unit_ids):
                proj = pc.get_projections(unit_id)
                assert proj.shape[1] == 5
            
            # test project_new
            unit_id = 3
            new_wfs = we.get_waveforms(unit_id)
            new_proj = pc.project_new(new_wfs, unit_id)
            assert new_proj.shape == (len(new_wfs), 5)

    def test_project_new(self):
        from sklearn.decomposition import IncrementalPCA

        we = self.we1
        if we.is_extension("principal_components"):
            we.delete_extension("principal_components")
        we_cp = we.select_units(we.unit_ids, self.cache_folder / 'toy_waveforms_1seg_cp')


        wfs0 = we.get_waveforms(unit_id=we.unit_ids[0])
        n_samples = wfs0.shape[1]
        n_channels = wfs0.shape[2]
        n_components = 5

        # local
        pc_local = compute_principal_components(we, n_components=n_components,
                                                load_if_exists=True, mode="by_channel_local")
        pc_local_par = compute_principal_components(we_cp, n_components=n_components,
                                                    load_if_exists=True, mode="by_channel_local",
                                                    n_jobs=2, progress_bar=True)

        all_pca = pc_local.get_pca_model()
        all_pca_par = pc_local_par.get_pca_model()

        assert len(all_pca) == we.get_num_channels()
        assert len(all_pca_par) == we.get_num_channels()

        for (pc, pc_par) in zip(all_pca, all_pca_par):
            assert np.allclose(pc.components_, pc_par.components_)

        # project
        new_waveforms = np.random.randn(100, n_samples, n_channels)
        new_proj = pc_local.project_new(new_waveforms)

        assert new_proj.shape == (100, n_components, n_channels)

        # global
        we.delete_extension('principal_components')
        pc_global = compute_principal_components(we, n_components=n_components,
                                                 load_if_exists=True, mode="by_channel_global")

        all_pca = pc_global.get_pca_model()
        assert isinstance(all_pca, IncrementalPCA)

        # project
        new_waveforms = np.random.randn(100, n_samples, n_channels)
        new_proj = pc_global.project_new(new_waveforms)

        assert new_proj.shape == (100, n_components, n_channels)

        # concatenated
        we.delete_extension('principal_components')
        pc_concatenated = compute_principal_components(we, n_components=n_components,
                                                       load_if_exists=True, mode="concatenated")

        all_pca = pc_concatenated.get_pca_model()
        assert isinstance(all_pca, IncrementalPCA)

        # project
        new_waveforms = np.random.randn(100, n_samples, n_channels)
        new_proj = pc_concatenated.project_new(new_waveforms)

        assert new_proj.shape == (100, n_components)


if __name__ == '__main__':
    test = PrincipalComponentsExtensionTest()
    test.setUp()
    # test.test_extension()
    # test.test_shapes()
    # test.test_compute_for_all_spikes()
    # test.test_sparse()
    test.test_project_new()
