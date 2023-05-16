import pytest
import numpy as np

from spikeinterface import NumpySorting
from spikeinterface import download_dataset
from spikeinterface import extract_waveforms

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks, clustering_methods

from spikeinterface.core import get_noise_levels
from spikeinterface.extractors import read_mearec

import time

def test_find_cluster_from_peaks():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording, gt_sorting = read_mearec(local_path)
    
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)
    
    noise_levels = get_noise_levels(recording, return_scaled=False)
    
    peaks = detect_peaks(recording, method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, exclude_sweep_ms=0.1,
                         verbose=False, noise_levels=noise_levels,
                         **job_kwargs)

    peak_locations = localize_peaks(recording, peaks, method='center_of_mass',
                                                            **job_kwargs)
    
    for method in clustering_methods.keys():
        method_kwargs = {}
        if method in ('position', 'position_and_pca'):
            method_kwargs['peak_locations'] = peak_locations
        if method in  ('sliding_hdbscan', 'position_and_pca'):
            method_kwargs['waveform_mode'] = 'shared_memory'
        
        t0 = time.perf_counter()
        labels, peak_labels = find_cluster_from_peaks(recording, peaks, method=method, method_kwargs=method_kwargs)
        t1 = time.perf_counter()
        print(method, 'found', len(labels), 'clusters in ',t1 - t0)



if __name__ == '__main__':
    test_find_cluster_from_peaks()
