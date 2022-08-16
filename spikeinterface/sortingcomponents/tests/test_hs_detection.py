import pytest

from spikeinterface import download_dataset
from spikeinterface.extractors import MEArecRecordingExtractor
from spikeinterface.sortingcomponents.hs_detection import run_hs_detection


def test_hs_detection():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    det = run_hs_detection(recording, save_shape=False)

    assert len(det) == 1
    assert set(det[0].keys()) == {'sample_ind', 'channel_ind',
                                  'amplitude', 'location'}
    assert det[0]['sample_ind'].shape[0] // 100 == 7  # around 700 by impl


if __name__ == '__main__':
    test_hs_detection()
