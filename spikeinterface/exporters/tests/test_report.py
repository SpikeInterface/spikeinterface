import unittest
import shutil
from pathlib import Path

import pytest

from spikeinterface import extract_waveforms, download_dataset
import spikeinterface.extractors as se
from spikeinterface.exporters import export_report


def test_export_report():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording, sorting = se.read_mearec(local_path)
    
    waveform_folder = Path('waveforms')
    output_folder= Path('mearec_GT_report')
    
    for f in (waveform_folder, output_folder):
        if f.is_dir():
            shutil.rmtree(f)
    
    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)

    export_report(waveform_extractor, output_folder)



if __name__ == '__main__':
    test_export_report()
