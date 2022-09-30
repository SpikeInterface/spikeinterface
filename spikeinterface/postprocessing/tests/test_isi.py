from typing import List
import numpy as np
import pytest 
from pathlib import Path

from spikeinterface import download_dataset, extract_waveforms, WaveformExtractor
import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_isi_histograms, ISIHistogramsCalculator

try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


def _test_ISI(sorting, window_ms: float, bin_ms: float, methods: List[str]):
	for method in methods:
		ISI, bins = compute_isi_histograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method=method)

		if method == "numpy":
			ref_ISI = ISI
			ref_bins = bins
		else:
			assert np.all(ISI == ref_ISI), f"Failed with method={method}"
			assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"

def test_compute_ISI():
	methods = ["numpy", "auto"]
	if HAVE_NUMBA:
		methods.append("numba")

	recording, sorting = se.toy_example(num_segments=2, num_units=10, duration=100)

	_test_ISI(sorting, window_ms=60.0, bin_ms=1.0, methods=methods)
	_test_ISI(sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


def test_isi_histogram_extension():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)

    folder = cache_folder / 'mearec_waveforms_locs'

    we = extract_waveforms(recording, sorting, folder,
                           ms_before=1., ms_after=2., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000, load_if_exists=False,
                           overwrite=True)
    isis, bins = compute_isi_histograms(we)
    
    # reload as an extension from we
    assert ISIHistogramsCalculator in we.get_available_extensions()
    assert we.is_extension('isi_histograms')
    isic = we.load_extension('isi_histograms')
    assert isinstance(isic, ISIHistogramsCalculator)
    assert 'isi_histograms' in isic._extension_data
    isic = ISIHistogramsCalculator.load_from_folder(folder)
    assert 'isi_histograms' in isic._extension_data

    # in-memory
    we_mem = extract_waveforms(we.recording, we.sorting, mode="memory")
    isis, bins = compute_isi_histograms(we_mem)

    # reload as an extension from we
    assert ISIHistogramsCalculator in we_mem.get_available_extensions()
    assert we_mem.is_extension('isi_histograms')
    isic = we.load_extension('isi_histograms')
    assert isinstance(isic, ISIHistogramsCalculator)
    assert 'isi_histograms' in isic._extension_data


if __name__ == '__main__':
    test_compute_ISI()
