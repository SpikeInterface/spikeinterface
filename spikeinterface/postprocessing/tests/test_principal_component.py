import pytest
import shutil
from pathlib import Path

import numpy as np

from spikeinterface import extract_waveforms, WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.postprocessing import WaveformPrincipalComponent, compute_principal_components


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"


def setup_module():
    for folder_name in ('toy_rec_1seg', 'toy_sorting_1seg', 'toy_waveforms_1seg',
                        'toy_rec_2seg', 'toy_sorting_2seg', 'toy_waveforms_2seg',
                        'toy_waveforms_1seg_filt'):
        if (cache_folder / folder_name).is_dir():
            shutil.rmtree(cache_folder / folder_name)

    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder=cache_folder / 'toy_rec_2seg')
    sorting = sorting.save(folder=cache_folder / 'toy_sorting_2seg')
    we = extract_waveforms(recording, sorting, cache_folder / 'toy_waveforms_2seg',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)

    recording, sorting = toy_example(
        num_segments=1, num_units=10, num_channels=12)
    recording = recording.save(folder=cache_folder / 'toy_rec_1seg')
    sorting = sorting.save(folder=cache_folder / 'toy_sorting_1seg')
    we = extract_waveforms(recording, sorting, cache_folder / 'toy_waveforms_1seg',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)
    shutil.copytree(cache_folder / 'toy_waveforms_1seg', cache_folder / 'toy_waveforms_1seg_cp')


def test_WaveformPrincipalComponent():
    we = WaveformExtractor.load_from_folder(cache_folder / 'toy_waveforms_2seg')
    unit_ids = we.sorting.unit_ids
    num_channels = we.recording.get_num_channels()
    pc = WaveformPrincipalComponent(we)

    for mode in ('by_channel_local', 'by_channel_global'):
        pc.set_params(n_components=5, mode=mode)
        print(pc)
        pc.run()
        for i, unit_id in enumerate(unit_ids):
            proj = pc.get_projections(unit_id)
            # print(comp.shape)
            assert proj.shape[1:] == (5, 4)

        # import matplotlib.pyplot as plt
        # cmap = plt.get_cmap('jet', len(unit_ids))
        # fig, axs = plt.subplots(ncols=num_channels)
        # for i, unit_id in enumerate(unit_ids):
        # comp = pca.get_components(unit_id)
        # print(comp.shape)
        # for chan_ind in range(num_channels):
        # ax = axs[chan_ind]
        # ax.scatter(comp[:, 0, chan_ind], comp[:, 1, chan_ind], color=cmap(i))
        # plt.show()

    for mode in ('concatenated',):
        pc.set_params(n_components=5, mode=mode)
        print(pc)
        pc.run()
        for i, unit_id in enumerate(unit_ids):
            proj = pc.get_projections(unit_id)
            assert proj.shape[1] == 5
            # print(comp.shape)

    all_labels, all_components = pc.get_all_projections()

    # relod as an extension from we
    assert WaveformPrincipalComponent in we.get_available_extensions()
    assert we.is_extension('principal_components')
    pc = we.load_extension('principal_components')
    assert isinstance(pc, WaveformPrincipalComponent)
    pc = WaveformPrincipalComponent.load_from_folder(
        cache_folder / 'toy_waveforms_2seg')

    # import matplotlib.pyplot as plt
    # cmap = plt.get_cmap('jet', len(unit_ids))
    # fig, ax = plt.subplots()
    # for i, unit_id in enumerate(unit_ids):
    # comp = pca.get_components(unit_id)
    # print(comp.shape)
    # ax.scatter(comp[:, 0], comp[:, 1], color=cmap(i))
    # plt.show()


def test_compute_principal_components_for_all_spikes():
    we = WaveformExtractor.load_from_folder(
        cache_folder / 'toy_waveforms_1seg')
    pc = compute_principal_components(we, load_if_exists=True)
    print(pc)

    pc_file1 = cache_folder / 'all_pc1.npy'
    pc.run_for_all_spikes(
        pc_file1, max_channels_per_template=7, chunk_size=10000, n_jobs=1)
    all_pc1 = np.load(pc_file1)

    pc_file2 = cache_folder / 'all_pc2.npy'
    pc.run_for_all_spikes(
        pc_file2, max_channels_per_template=7, chunk_size=10000, n_jobs=2)
    all_pc2 = np.load(pc_file2)

    assert np.array_equal(all_pc1, all_pc2)


def test_pca_models_and_project_new():
    from sklearn.decomposition import IncrementalPCA
    if (cache_folder / "toy_waveforms_1seg" / "principal_components").is_dir():
        shutil.rmtree(cache_folder / "toy_waveforms_1seg" /
                      "principal_components")
    we = WaveformExtractor.load_from_folder(
        cache_folder / 'toy_waveforms_1seg')
    we_cp = WaveformExtractor.load_from_folder(
        cache_folder / 'toy_waveforms_1seg_cp')

    wfs0 = we.get_waveforms(unit_id=we.sorting.unit_ids[0])
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

    assert len(all_pca) == we.recording.get_num_channels()
    assert len(all_pca_par) == we.recording.get_num_channels()

    for (pc, pc_par) in zip(all_pca, all_pca_par):
        assert np.allclose(pc.components_, pc_par.components_)

    # project
    new_waveforms = np.random.randn(100, n_samples, n_channels)
    new_proj = pc_local.project_new(new_waveforms)

    assert new_proj.shape == (100, n_components, n_channels)

    # global
    if (cache_folder / "toy_waveforms_1seg" / "principal_components").is_dir():
        shutil.rmtree(cache_folder / "toy_waveforms_1seg" /
                      "principal_components")

    pc_global = compute_principal_components(we, n_components=n_components,
                                             load_if_exists=True, mode="by_channel_global")

    all_pca = pc_global.get_pca_model()
    assert isinstance(all_pca, IncrementalPCA)

    # project
    new_waveforms = np.random.randn(100, n_samples, n_channels)
    new_proj = pc_global.project_new(new_waveforms)

    assert new_proj.shape == (100, n_components, n_channels)

    # concatenated
    if Path(cache_folder / "toy_waveforms_1seg" / "principal_components").is_dir():
        shutil.rmtree(cache_folder / "toy_waveforms_1seg" /
                      "principal_components")

    pc_concatenated = compute_principal_components(we, n_components=n_components,
                                                   load_if_exists=True, mode="concatenated")

    all_pca = pc_concatenated.get_pca_model()
    assert isinstance(all_pca, IncrementalPCA)

    # project
    new_waveforms = np.random.randn(100, n_samples, n_channels)
    new_proj = pc_concatenated.project_new(new_waveforms)

    assert new_proj.shape == (100, n_components)


def test_select_units():
    we = WaveformExtractor.load_from_folder(
        cache_folder / 'toy_waveforms_1seg')
    pc = compute_principal_components(we, load_if_exists=True)

    keep_units = we.sorting.get_unit_ids()[::2]
    we_filt = we.select_units(
        keep_units, cache_folder / 'toy_waveforms_1seg_filt')
    assert "principal_components" in we_filt.get_available_extension_names()


if __name__ == '__main__':
    setup_module()
    #~ test_WaveformPrincipalComponent()
    #~ test_compute_principal_components_for_all_spikes()
    test_pca_models_and_project_new()
