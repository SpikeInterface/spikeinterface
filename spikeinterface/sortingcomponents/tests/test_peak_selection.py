import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.sortingcomponents import detect_peaks, select_peaks
from spikeinterface.toolkit import get_noise_levels

from spikeinterface.extractors import MEArecRecordingExtractor


def test_detect_peaks():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    # by_channel

    noise_levels = get_noise_levels(recording, return_scaled=False)

    peaks = detect_peaks(recording, method='by_channel',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=False, noise_levels=noise_levels)

    subset_peaks = select_peaks(peaks, max_peaks_per_channel=100)
    subset_peaks = select_peaks(
        peaks, 'smart_sampling', max_peaks_per_channel=100, noise_levels=noise_levels)

    assert len(subset_peaks) < len(peaks)


if __name__ == '__main__':
    test_detect_peaks()
