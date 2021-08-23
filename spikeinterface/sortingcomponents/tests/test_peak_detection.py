import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.sortingcomponents import detect_peaks

from spikeinterface.extractors import MEArecRecordingExtractor


def test_detect_peaks():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    peaks = detect_peaks(recording,
                         method='by_channel',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=False,
                         outputs='numpy_compact'
                         )

    sample_inds, chan_inds, amplitudes, seg_inds = detect_peaks(recording,
                                                                method='locally_exclusive',
                                                                peak_sign='neg', detect_threshold=5, n_shifts=2,
                                                                chunk_size=10000, verbose=1, progress_bar=False,
                                                                outputs='numpy_split'
                                                                )

    # DEBUG
    # import matplotlib.pyplot as plt
    # import spikeinterface.widgets as sw
    # chan_offset = 500
    # traces = recording.get_traces()
    # traces += np.arange(traces.shape[1])[None, :] * chan_offset
    # print(traces.shape)
    # fig, ax = plt.subplots()
    # ax.plot(traces, color='k')
    # ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, color='r')
    # plt.show()


if __name__ == '__main__':
    test_detect_peaks()
