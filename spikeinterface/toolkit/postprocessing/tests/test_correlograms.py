import pytest
import numpy as np

from spikeinterface import download_dataset, extract_waveforms
import spikeinterface.extractors as se
from spikeinterface.toolkit import compute_autocorrelogram_from_spiketrain, compute_correlograms

try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


def test_compute_correlograms():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    # ~ recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    unit_ids = sorting.unit_ids
    sorting2 = sorting.select_units(unit_ids[:3])
    correlograms, bins = compute_correlograms(sorting2)


@pytest.mark.skipif(not HAVE_NUMBA, reason="requires numba")
def test_compute_autocorrelogram_from_spiketrain():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    # ~ recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)
    fs = sorting.get_sampling_frequency()

    spike_train = sorting.get_unit_spike_train(sorting.get_unit_ids()[0])
    correlograms, bins = compute_autocorrelogram_from_spiketrain(spike_train, max_time=int(35.0*fs), bin_size=int(0.5*fs), sampling_f=fs)


if __name__ == '__main__':
    test_compute_correlograms()

    if HAVE_NUMBA:
        test_compute_autocorrelogram_from_spiketrain()
