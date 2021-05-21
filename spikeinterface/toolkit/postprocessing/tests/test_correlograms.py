import pytest
import numpy as np

from spikeinterface import download_dataset, extract_waveforms
import spikeinterface.extractors as se
from spikeinterface.toolkit import compute_correlograms
    
    
def test_compute_correlograms():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    #~ recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)
    
    unit_ids = sorting.unit_ids
    sorting2 = sorting.select_units(unit_ids[:3])
    correlograms, bins = compute_correlograms(sorting2)
    

if __name__ == '__main__':
    test_compute_correlograms()
