from typing import List
import pytest
import numpy as np
from pathlib import Path

from spikeinterface import download_dataset, extract_waveforms, WaveformExtractor
import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_correlograms, CorrelogramsCalculator

try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


def _test_correlograms(sorting, window_ms: float, bin_ms: float, methods: List[str]):
    for method in methods:
        correlograms, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, symmetrize=True, 
                                                  method=method)

        if method == "numpy":
            ref_correlograms = correlograms
            ref_bins = bins
        else:
            assert np.all(correlograms == ref_correlograms), f"Failed with method={method}"
            assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"

@pytest.mark.skip(reason="Is going to be fixed (PR #750)")
def test_compute_correlograms():
    methods = ["numpy", "auto"]
    if HAVE_NUMBA:
        methods.append("numba")

    recording, sorting = se.toy_example(num_segments=2, num_units=10, duration=100)

    _test_correlograms(sorting, window_ms=60.0, bin_ms=2.0, methods=methods)
    _test_correlograms(sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


def test_correlograms_extension():
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
    correlograms, bins = compute_correlograms(we, method='numpy')
    
    # reload as an extension from we
    assert CorrelogramsCalculator in we.get_available_extensions()
    assert we.is_extension('correlograms')
    ccc = we.load_extension('correlograms')
    assert isinstance(ccc, CorrelogramsCalculator)
    assert 'ccgs' in ccc._extension_data
    ccc = CorrelogramsCalculator.load_from_folder(folder)
    assert 'ccgs' in ccc._extension_data

    # in-memory
    we_mem = extract_waveforms(we.recording, we.sorting, mode="memory")
    isis, bins = compute_correlograms(we_mem)

    # reload as an extension from we
    assert CorrelogramsCalculator in we_mem.get_available_extensions()
    assert we_mem.is_extension('correlograms')
    ccc = we.load_extension('correlograms')
    assert isinstance(ccc, CorrelogramsCalculator)
    assert 'ccgs' in ccc._extension_data

if __name__ == '__main__':
    test_compute_correlograms()

