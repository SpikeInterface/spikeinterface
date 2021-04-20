import unittest
import shutil
from pathlib import Path

import pytest



from spikeinterface import extract_waveforms, download_dataset
import spikeinterface.extractors as se
from spikeinterface.toolkit.postprocessing import export_to_phy


def test_export_to_phy():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    
    
    waveform_folder = Path('waveforms')
    output_folder= Path('phy_output')
    
    for f in (waveform_folder, output_folder):
        if f.is_dir():
            shutil.rmtree(f)
    
    
    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)

    export_to_phy(recording, sorting, output_folder, waveform_extractor,
                  compute_pc_features=False,
                  compute_amplitudes=True)
    
    


if __name__ == '__main__':
    test_export_to_phy()
