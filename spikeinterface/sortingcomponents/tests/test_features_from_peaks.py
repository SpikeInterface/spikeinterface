import pytest
import numpy as np

from spikeinterface import download_dataset, BaseSorting
from spikeinterface.extractors import MEArecRecordingExtractor

from spikeinterface.sortingcomponents.features_from_peaks import compute_features_from_peaks

from spikeinterface.toolkit import get_noise_levels

from spikeinterface.sortingcomponents.peak_detection import detect_peaks


def test_features_from_peaks():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    noise_levels = get_noise_levels(recording, return_scaled=False)

    peaks = detect_peaks(recording, method='by_channel',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=False, noise_levels=noise_levels)

    # locally_exclusive
    features = compute_features_from_peaks(recording, peaks, ['ptp', 'energy', 'com', 'dist_com_vs_max_ptp_channel', 'dist_com_vs_max_std_channel'])

    print(features)

if __name__ == '__main__':
    test_features_from_peaks()
