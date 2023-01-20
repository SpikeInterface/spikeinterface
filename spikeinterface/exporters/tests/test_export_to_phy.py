import pytest
import shutil
from pathlib import Path

import numpy as np

from spikeinterface import extract_waveforms, download_dataset, compute_sparsity
import spikeinterface.extractors as se
from spikeinterface.exporters import export_to_phy
from spikeinterface.postprocessing import compute_principal_components

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "exporters"
else:
    cache_folder = Path("cache_folder") / "exporters"


def test_export_to_phy():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    waveform_folder = cache_folder / 'waveforms'
    output_folder = cache_folder / 'phy_output'

    for f in (waveform_folder, output_folder):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)

    export_to_phy(waveform_extractor, output_folder,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  n_jobs=1, chunk_size=10000,
                  progress_bar=True)


def test_export_to_phy_by_property():
    num_units = 4
    recording, sorting = se.toy_example(num_channels=8, duration=10, num_units=num_units, num_segments=1)
    recording.set_channel_groups([0, 0, 0, 0, 1, 1, 1, 1])
    sorting.set_property("group", [0, 0, 1, 1])

    waveform_folder = cache_folder / 'waveforms'
    waveform_folder_rm = cache_folder / 'waveforms_rm'
    output_folder = cache_folder / 'phy_output'
    output_folder_rm = cache_folder / 'phy_output_rm'
    rec_folder = cache_folder / 'rec'
    sort_folder = cache_folder / 'sort'

    for f in (waveform_folder, waveform_folder_rm, output_folder, output_folder_rm, rec_folder, sort_folder):
        if f.is_dir():
            shutil.rmtree(f)

    recording = recording.save(folder=rec_folder)
    sorting = sorting.save(folder=sort_folder)

    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)
    sparsity_group = compute_sparsity(waveform_extractor, method="by_property",
                                      by_property="group")
    export_to_phy(waveform_extractor, output_folder,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  sparsity=sparsity_group,
                  n_jobs=1, chunk_size=10000, progress_bar=True)

    template_inds = np.load(output_folder / "template_ind.npy")
    assert template_inds.shape == (num_units, 4)

    # Remove one channel
    recording_rm = recording.channel_slice([0, 2, 3, 4, 5, 6, 7])
    waveform_extractor_rm = extract_waveforms(recording_rm, sorting, waveform_folder_rm)
    sparsity_group = compute_sparsity(waveform_extractor_rm, method="by_property",
                                      by_property="group")

    export_to_phy(waveform_extractor_rm, output_folder_rm,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  sparsity=sparsity_group,
                  n_jobs=1, chunk_size=10000,
                  progress_bar=True)

    template_inds = np.load(output_folder_rm / "template_ind.npy")
    assert template_inds.shape == (num_units, 4)
    assert len(np.where(template_inds == -1)[0]) > 0


def test_export_to_phy_by_sparsity():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    waveform_folder = cache_folder / 'waveforms'
    output_folder_radius = cache_folder / 'phy_output_radius'
    output_folder_multi_sparse = cache_folder / 'phy_output_multi_sparse'

    for f in (waveform_folder, output_folder_radius, output_folder_multi_sparse):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)
    sparsity_radius = compute_sparsity(waveform_extractor, method="radius", radius_um=50.)
    export_to_phy(waveform_extractor, output_folder_radius,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  sparsity=sparsity_radius,
                  n_jobs=1, chunk_size=10000, progress_bar=True)

    template_ind = np.load(output_folder_radius / "template_ind.npy")
    pc_ind = np.load(output_folder_radius / "pc_feature_ind.npy")
    # templates have different shapes!
    assert -1 in template_ind
    assert -1 in pc_ind

    # pre-compute PC with another sparsity
    sparsity_radius_small = compute_sparsity(waveform_extractor, method="radius", radius_um=30.)
    pc = compute_principal_components(waveform_extractor, sparsity=sparsity_radius_small)
    export_to_phy(waveform_extractor, output_folder_multi_sparse,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  sparsity=sparsity_radius,
                  n_jobs=1, chunk_size=10000,
                  progress_bar=True)

    template_ind = np.load(output_folder_multi_sparse / "template_ind.npy")
    pc_ind = np.load(output_folder_multi_sparse / "pc_feature_ind.npy")
    # templates have different shapes!
    assert -1 in template_ind
    assert -1 in pc_ind
    # PC sparsity is more stringent than teplate sparsity
    assert pc_ind.shape[1] < template_ind.shape[1]


if __name__ == '__main__':
    test_export_to_phy()
    test_export_to_phy_by_property()
    test_export_to_phy_by_sparsity()
