import pytest
import numpy as np
from pathlib import Path
import shutil

from spikeinterface import download_dataset, extract_waveforms, WaveformExtractor
import spikeinterface.extractors as se
from spikeinterface.toolkit import compute_spike_amplitudes, SpikeAmplitudesCalculator


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"


def test_compute_spike_amplitudes():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    folder = cache_folder / 'mearec_waveforms_amps'

    we = extract_waveforms(recording, sorting, folder,
                           ms_before=1., ms_after=2., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000, load_if_exists=False,
                           overwrite=True)

    amplitudes = compute_spike_amplitudes(
        we, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=1)
    amplitudes = compute_spike_amplitudes(
        we, peak_sign='neg', outputs='by_unit', chunk_size=10000, n_jobs=1)

    gain = 0.1
    recording.set_channel_gains(gain)
    recording.set_channel_offsets(0)

    folder = cache_folder / 'mearec_waveforms_amps_scaled'

    we_scaled = extract_waveforms(recording, sorting, folder,
                                  ms_before=1., ms_after=2., max_spikes_per_unit=500,
                                  n_jobs=1, chunk_size=30000, load_if_exists=False,
                                  overwrite=True, return_scaled=True)

    amplitudes_scaled = compute_spike_amplitudes(we_scaled, peak_sign='neg', outputs='concatenated', chunk_size=10000, n_jobs=1,
                                                 return_scaled=True)
    amplitudes_unscaled = compute_spike_amplitudes(we_scaled, peak_sign='neg', outputs='concatenated', chunk_size=10000,
                                                   n_jobs=1, return_scaled=False)

    assert np.allclose(amplitudes_scaled[0], amplitudes_unscaled[0] * gain)

    # reload as an extension from we
    assert SpikeAmplitudesCalculator in we.get_available_extensions()
    assert we_scaled.is_extension('spike_amplitudes')
    sac = we.load_extension('spike_amplitudes')
    assert isinstance(sac, SpikeAmplitudesCalculator)
    assert sac._amplitudes is not None
    qmc = SpikeAmplitudesCalculator.load_from_folder(folder)
    assert sac._amplitudes is not None


def test_compute_spike_amplitudes_parallel():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    folder = cache_folder / 'mearec_waveforms_all_amps'

    we = extract_waveforms(recording, sorting, folder,
                           ms_before=1., ms_after=2., max_spikes_per_unit=None,
                           n_jobs=1, chunk_size=30000, load_if_exists=True)

    amplitudes1 = compute_spike_amplitudes(
        we, peak_sign='neg', load_if_exists=False, outputs='concatenated', chunk_size=10000, n_jobs=1)
    # TODO : fix multi processing for spike amplitudes!!!!!!!
    amplitudes2 = compute_spike_amplitudes(
        we, peak_sign='neg', load_if_exists=False, outputs='concatenated', chunk_size=10000, n_jobs=2)

    assert np.array_equal(amplitudes1[0], amplitudes2[0])
    # shutil.rmtree(folder)


def test_select_units():
    we = WaveformExtractor.load_from_folder(cache_folder / 'mearec_waveforms_amps')
    amps = compute_spike_amplitudes(we, load_if_exists=True)

    keep_units = we.sorting.get_unit_ids()[::2]
    we_filt = we.select_units(
        keep_units, cache_folder / 'mearec_waveforms_filt_amps')
    assert "spike_amplitudes" in we_filt.get_available_extension_names()


if __name__ == '__main__':
    test_compute_spike_amplitudes()
    test_compute_spike_amplitudes_parallel()
