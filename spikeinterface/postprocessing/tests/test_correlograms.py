from typing import List
import pytest
import numpy as np
from pathlib import Path

from spikeinterface import download_dataset, extract_waveforms,  NumpySorting
import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_correlograms, CrossCorrelogramsCalculator
from spikeinterface.postprocessing.correlograms import _make_bins
from spikeinterface.core.testing_tools import generate_sorting


try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "postprocessing"
else:
    cache_folder = Path("cache_folder") / "postprocessing"


def test_make_bins():
    sorting = generate_sorting(num_units=5, sampling_frequency=30000., durations=[10.325, 3.5])
    
    window_ms = 43.57
    bin_ms = 1.6421
    bins, window_size, bin_size = _make_bins(sorting, window_ms, bin_ms)
    assert bins.size == np.floor(window_ms / bin_ms) +1 
    # print(bins, window_size, bin_size)

    window_ms=60.0
    bin_ms=2.0
    bins, window_size, bin_size = _make_bins(sorting, window_ms, bin_ms)
    assert bins.size == np.floor(window_ms / bin_ms) +1 
    # print(bins, window_size, bin_size)
    

def _test_correlograms(sorting, window_ms, bin_ms, methods):
    for method in methods:
        correlograms, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, symmetrize=True, 
                                                  method=method)
        if method == "numpy":
            ref_correlograms = correlograms
            ref_bins = bins
        else:
            import matplotlib.pyplot as plt
            
            for i in range(ref_correlograms.shape[1]):
                fig, ax = plt.subplots()
                ax.plot(bins[:-1], ref_correlograms[0, i, :], color='green', label='numpy')
                ax.plot(bins[:-1], correlograms[0, i, :], color='red', label=method)
                ax.legend()

                plt.show()
            assert np.all(correlograms == ref_correlograms), f"Failed with method={method}"
            assert np.allclose(bins, ref_bins, atol=1e-10), f"Failed with method={method}"

def test_equal_results_correlograms():
    # compare that the 2 methods have same results
    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

    sorting = generate_sorting(num_units=5, sampling_frequency=30000., durations=[10.325, 3.5])

    _test_correlograms(sorting, window_ms=60.0, bin_ms=2.0, methods=methods)
    _test_correlograms(sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


def test_flat_cross_correlogram():
    sorting = generate_sorting(num_units=2, sampling_frequency=10000., durations=[100000.])

    methods = ["numpy"]
    if HAVE_NUMBA:
        methods.append("numba")

    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    
    for method in methods:
        correlograms, bins = compute_correlograms(sorting, window_ms=50., bin_ms=0.1, method=method)
        cc = correlograms[1, 0, :].copy()
        m = np.mean(cc)
        assert np.all(cc > (m*0.90))
        assert np.all(cc < (m*1.10))

        #~ ax.plot(bins[:-1], cc, label=method)
    #~ ax.legend()
    #~ ax.set_ylim(0, np.max(correlograms) * 1.1)
    #~ plt.show()



def test_auto_cross_correlograms():
    # check if cross correlogram is the same as autocorrelogram
    # by removing n spike in bin zeros
    # this is not the case for numpy method
    
    methods = ['numpy']
    #~ methods = []
    if HAVE_NUMBA:
        methods.append("numba")
    print(methods)
    
    num_spike = 2000
    spike_times = np.sort(np.unique(np.random.randint(0, 100000, num_spike)))
    num_spike = spike_times.size
    units_dict = {'1': spike_times, '2': spike_times}
    sorting = NumpySorting.from_dict([units_dict], sampling_frequency=10000.)
    

    for method in methods:
        correlograms, bins = compute_correlograms(sorting, window_ms=10., bin_ms=1., method=method)
        
        num_half_bins = correlograms.shape[2]  // 2

        cc = correlograms[1, 0, :].copy()
        ac = correlograms[0, 0, :]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(bins[:-1], cc, marker='*',  color='red', label='cross-corr')
        ax.plot(bins[:-1], ac, marker='*', color='green', label='auto-corr')
        ax.set_title(method)
        ax.legend()
        ax.set_ylim(0, np.max(correlograms) * 1.1)
        plt.show()
        
        cc[num_half_bins]  -= num_spike
        # assert np.array_equal(cc, ac)
        



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
    assert CrossCorrelogramsCalculator in we.get_available_extensions()
    assert we.is_extension('crosscorrelograms')
    ccc = we.load_extension('crosscorrelograms')
    assert isinstance(ccc, CrossCorrelogramsCalculator)
    assert ccc.ccgs is not None
    ccc = CrossCorrelogramsCalculator.load_from_folder(folder)
    assert ccc.ccgs is not None


if __name__ == '__main__':
    #~ test_make_bins()
    test_equal_results_correlograms()
    #~ test_flat_cross_correlogram()
    #~ test_auto_cross_correlograms()
    
    
    #~ test_correlograms_extension()

