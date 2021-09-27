import unittest
import shutil
from pathlib import Path

import numpy as np

from spikeinterface import extract_waveforms, download_dataset
import spikeinterface.extractors as se
from spikeinterface.exporters import export_to_phy

def setup_module():
    for folder in ('waveforms', 'waveforms_rm', 'phy_output', 'phy_output_rm', 'rec', 'sort'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)


def test_export_to_phy():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    waveform_folder = Path('waveforms')
    output_folder = Path('phy_output')

    for f in (waveform_folder, output_folder):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)

    export_to_phy(waveform_extractor, output_folder,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  max_channels_per_template=8,
                  n_jobs=1, chunk_size=10000, progress_bar=True)


def test_export_to_phy_by_property():
    num_units = 4
    recording, sorting = se.toy_example(num_channels=8, duration=10, num_units=num_units, num_segments=1)
    recording.set_channel_groups([0, 0, 0, 0, 1, 1, 1, 1])
    sorting.set_property("group", [0, 0, 1, 1])

    waveform_folder = Path('waveforms')
    waveform_folder_rm = Path('waveforms_rm')
    output_folder = Path('phy_output')
    output_folder_rm = Path('phy_output_rm')
    rec_folder = Path("rec")
    sort_folder = Path("sort")

    for f in (waveform_folder, waveform_folder_rm, output_folder, output_folder_rm, rec_folder, sort_folder):
        if f.is_dir():
            shutil.rmtree(f)

    recording = recording.save(folder=rec_folder)
    sorting = sorting.save(folder=sort_folder)

    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)

    export_to_phy(waveform_extractor, output_folder,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  max_channels_per_template=8,
                  sparsity_dict=dict(method="by_property", by_property="group"),
                  n_jobs=1, chunk_size=10000, progress_bar=True)

    template_inds = np.load(output_folder / "template_ind.npy")
    assert template_inds.shape == (num_units, 4)

    # Remove one channel
    recording_rm = recording.channel_slice([0, 2, 3, 4, 5, 6, 7])
    waveform_extractor_rm = extract_waveforms(recording_rm, sorting, waveform_folder_rm)

    export_to_phy(waveform_extractor_rm, output_folder_rm,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  max_channels_per_template=8,
                  sparsity_dict=dict(method="by_property", by_property="group"),
                  n_jobs=1, chunk_size=10000, progress_bar=True)

    template_inds = np.load(output_folder_rm / "template_ind.npy")
    assert template_inds.shape == (num_units, 4)
    assert len(np.where(template_inds == -1)[0]) > 0


def test_export_to_phy_by_sparsity():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    waveform_folder = Path('waveforms')
    output_folder_radius = Path('phy_output_radius')
    output_folder_thr = Path('phy_output_thr')

    for f in (waveform_folder, output_folder_radius, output_folder_thr):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)

    export_to_phy(waveform_extractor, output_folder_radius,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  max_channels_per_template=None,
                  sparsity_dict=dict(method="radius", radius_um=50),
                  n_jobs=1, chunk_size=10000, progress_bar=True)

    template_ind = np.load(output_folder_radius / "template_ind.npy")
    # templates have different shapes!
    assert -1 in template_ind

    export_to_phy(waveform_extractor, output_folder_thr,
                  compute_pc_features=True,
                  compute_amplitudes=True,
                  max_channels_per_template=None,
                  sparsity_dict=dict(method="threshold", threshold=2),
                  n_jobs=1, chunk_size=10000, progress_bar=True)

    template_ind = np.load(output_folder_thr / "template_ind.npy")
    # templates have different shapes!
    assert -1 in template_ind


if __name__ == '__main__':
    # test_export_to_phy_by_property()
    test_export_to_phy_by_sparsity()
