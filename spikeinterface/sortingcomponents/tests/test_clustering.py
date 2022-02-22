import pytest
import numpy as np

from spikeinterface import NumpySorting
from spikeinterface import download_dataset
from spikeinterface import extract_waveforms

from spikeinterface.sortingcomponents import detect_peaks
from spikeinterface.sortingcomponents import find_cluster_from_peaks, clustering_methods

from spikeinterface.toolkit import get_noise_levels
from spikeinterface.extractors import read_mearec


def test_find_cluster_from_peaks():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording, gt_sorting = read_mearec(local_path)
    
    
    noise_levels = get_noise_levels(recording, return_scaled=False)
    
    peaks = detect_peaks(recording, method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=False, progress_bar=False, noise_levels=noise_levels)
    # print(peaks)
    
    for method in clustering_methods:
        print(method)
    
        labels, peak_labels = find_cluster_from_peaks(recording, peaks, method=method)
        print(labels)
    
    
    
    
    
if __name__ == '__main__':
    test_find_cluster_from_peaks()