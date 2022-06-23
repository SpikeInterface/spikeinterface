import pytest
import numpy as np
from pathlib import Path
import shutil

from spikeinterface import download_dataset, extract_waveforms, WaveformExtractor
import spikeinterface.extractors as se
from spikeinterface.toolkit import compute_spike_locations, SpikeLocationsCalculator


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"


def test_compute_spike_locations():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    folder = cache_folder / 'mearec_waveforms'

    we = extract_waveforms(recording, sorting, folder,
                           ms_before=1., ms_after=2., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000, load_if_exists=False,
                           overwrite=True)

    # test all options
    locs_com = compute_spike_locations(
        we, method="center_of_mass", chunk_size=10000, n_jobs=1)
    locs_com = compute_spike_locations(
        we, method="center_of_mass", outputs='by_unit', chunk_size=10000, n_jobs=1)
    locs_mono = compute_spike_locations(
        we, method="monopolar_triangulation", chunk_size=10000, n_jobs=1)
    locs_mono = compute_spike_locations(
        we, method="monopolar_triangulation", outputs='by_unit', chunk_size=10000, n_jobs=1)
    

    # reload as an extension from we
    assert SpikeLocationsCalculator in we.get_available_extensions()
    assert we.is_extension('spike_locations')
    slc = we.load_extension('spike_locations')
    assert isinstance(slc, SpikeLocationsCalculator)
    assert slc._locations is not None
    slc = SpikeLocationsCalculator.load_from_folder(folder)
    assert slc._locations is not None


def test_compute_spike_locations_parallel():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    folder = cache_folder / 'mearec_waveforms_all'

    we = extract_waveforms(recording, sorting, folder,
                           ms_before=1., ms_after=2., max_spikes_per_unit=None,
                           n_jobs=1, chunk_size=30000, load_if_exists=True)

    locs_mono = compute_spike_locations(
        we, method="monopolar_triangulation", chunk_size=10000, n_jobs=1)
    locs_mono = compute_spike_locations(
        we, method="monopolar_triangulation", chunk_size=10000, n_jobs=2)
    
    assert np.array_equal(locs_mono[0], locs_mono[0])
    # shutil.rmtree(folder)


def test_select_units():
    we = WaveformExtractor.load_from_folder(cache_folder / 'mearec_waveforms')
    locs = compute_spike_locations(we, load_if_exists=True)

    keep_units = we.sorting.get_unit_ids()[::2]
    we_filt = we.select_units(
        keep_units, cache_folder / 'mearec_waveforms_filt')
    assert "spike_locations" in we_filt.get_available_extension_names()


if __name__ == '__main__':
    test_compute_spike_locations()
    test_compute_spike_locations_parallel()
