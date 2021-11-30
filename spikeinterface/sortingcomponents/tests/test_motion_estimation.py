import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.sortingcomponents import detect_peaks, estimate_motion, make_motion_histogram

from spikeinterface.extractors import MEArecRecordingExtractor


def test_estimate_motion():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)
    
    # detect and localize
    peaks = detect_peaks(recording,
                         method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=True,
                         localization_dict=dict(method='center_of_mass', local_radius_um=150, ms_before=0.1, ms_after=0.3),
                         #~ localization_dict=dict(method='monopolar_triangulation', local_radius_um=150, ms_before=0.1, ms_after=0.3, max_distance_um=1000),
                         )
    
    print(peaks)
        
    
    motion_histogram, temporal_bins, spatial_bins = make_motion_histogram(recording, peaks, weight_with_amplitude=True)
    print(motion_histogram.shape, temporal_bins.size, spatial_bins.size)
    
    print(motion_histogram)
    

    # DEBUG
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    extent = (temporal_bins[0], temporal_bins[-1], spatial_bins[0], spatial_bins[-1])
    im = ax.imshow(motion_histogram.T, interpolation='nearest',
                        origin='lower', aspect='auto', extent=extent)

    fig, ax = plt.subplots()
    ax.scatter(peaks['sample_ind'] / recording.get_sampling_frequency(),peaks['y'], color='r')
    plt.show()


if __name__ == '__main__':
    test_estimate_motion()
