import numpy as np
from pathlib import Path
import shutil

from spikeinterface import download_dataset, extract_waveforms
import spikeinterface.extractors as se
from spikeinterface.toolkit import get_spike_amplitudes


def test_get_spike_amplitudes():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    folder = Path('mearec_waveforms')

    we = extract_waveforms(recording, sorting, folder,
                           ms_before=1., ms_after=2., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000, load_if_exists=False,
                           overwrite=True)

    amplitudes = get_spike_amplitudes(we, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=1)
    amplitudes = get_spike_amplitudes(we, peak_sign='neg', outputs='by_unit', chunk_size=10000, n_jobs=1)

    # shutil.rmtree(folder)


def test_get_spike_amplitudes_parallel():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    folder = Path('mearec_waveforms_all')

    we = extract_waveforms(recording, sorting, folder,
                           ms_before=1., ms_after=2., max_spikes_per_unit=None,
                           n_jobs=1, chunk_size=30000, load_if_exists=True)

    amplitudes1 = get_spike_amplitudes(we, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=1)
    # amplitudes2 = get_spike_amplitudes(we, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=2)

    # assert np.array_equal(amplitudes1, amplitudes2)
    # shutil.rmtree(folder)


if __name__ == '__main__':
    test_get_spike_amplitudes()
    test_get_spike_amplitudes_parallel()

