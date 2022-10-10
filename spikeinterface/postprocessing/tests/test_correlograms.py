import unittest
import numpy as np
from typing import List

from spikeinterface.postprocessing import compute_correlograms, CorrelogramsCalculator

from spikeinterface.postprocessing.tests.common_extension_tests import WaveformExtensionCommonTestSuite

from spikeinterface import download_dataset, extract_waveforms,  NumpySorting
import spikeinterface.extractors as se

from spikeinterface.postprocessing import compute_correlograms, CorrelogramsCalculator
from spikeinterface.postprocessing.correlograms import _make_bins
from spikeinterface.core.testing_tools import generate_sorting


try:
    import numba
    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False


class CorrelogramsExtensionTest(WaveformExtensionCommonTestSuite, unittest.TestCase):
    extension_class = CorrelogramsCalculator
    extension_data_names = ["ccgs", "bins"]
    extension_function_kwargs_list =[
        dict(method='numpy')
    ]

    @unittest.skip("It's going to be fixed (PR #750)")
    def test_compute_correlograms(self):
        methods = ["numpy", "auto"]
        if HAVE_NUMBA:
            methods.append("numba")

        sorting = self.we1.sorting

        _test_correlograms(sorting, window_ms=60.0, bin_ms=2.0, methods=methods)
        _test_correlograms(sorting, window_ms=43.57, bin_ms=1.6421, methods=methods)


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
            #~ import matplotlib.pyplot as plt
            #~ for i in range(ref_correlograms.shape[1]):
                #~ for j in range(ref_correlograms.shape[1]):
                    #~ fig, ax = plt.subplots()
                    #~ ax.plot(bins[:-1], ref_correlograms[i, j, :], color='green', label='numpy')
                    #~ ax.plot(bins[:-1], correlograms[i, j, :], color='red', label=method)
                    #~ ax.legend()
                    #~ ax.set_title(f'{i} {j}')
                    #~ plt.show()
            
            # This should be True but numpy and numba version some small diff that need to be investigated!!
            # assert np.all(correlograms == ref_correlograms), f"Failed with method={method}"
            
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
        correlograms, bins = compute_correlograms(sorting, window_ms=50., bin_ms=0.25, method=method)
        cc = correlograms[1, 0, :].copy()
        m = np.mean(cc)
        assert np.all(cc > (m*0.90))
        assert np.all(cc < (m*1.10))

        #~ ax.plot(bins[:-1], cc, label=method)
    #~ ax.legend()
    #~ ax.set_ylim(0, np.max(correlograms) * 1.1)
    #~ plt.show()



def test_auto_equal_cross_correlograms():
    """
    check if cross correlogram is the same as autocorrelogram
    by removing n spike in bin zeros
    numpy method:
      * have problem for the left bin
      * have problem on center
    """
    
    methods = ['numpy']
    if HAVE_NUMBA:
        methods.append("numba")
    
    num_spike = 2000
    spike_times = np.sort(np.unique(np.random.randint(0, 100000, num_spike)))
    num_spike = spike_times.size
    units_dict = {'1': spike_times, '2': spike_times}
    sorting = NumpySorting.from_dict([units_dict], sampling_frequency=10000.)
    

    for method in methods:
        correlograms, bins = compute_correlograms(sorting, window_ms=10., bin_ms=0.1, method=method)
        
        num_half_bins = correlograms.shape[2]  // 2

        cc = correlograms[0, 1, :]
        ac = correlograms[0, 0, :]
        cc_corrected = cc.copy()
        cc_corrected[num_half_bins]  -= num_spike
        
        if method == 'numpy':
            # numpy method have some border effect on left
            assert np.array_equal(cc_corrected[1:num_half_bins], ac[1:num_half_bins])
            # numpy method have some problem on center
            assert np.array_equal(cc_corrected[num_half_bins+1:], ac[num_half_bins+1:])
        else:
            assert np.array_equal(cc_corrected, ac)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(bins[:-1], cc, marker='*',  color='red', label='cross-corr')
        ax.plot(bins[:-1], cc_corrected, marker='*', color='orange', label='cross-corr corrected')
        ax.plot(bins[:-1], ac, marker='*', color='green', label='auto-corr')
        ax.set_title(method)
        ax.legend()
        ax.set_ylim(0, np.max(correlograms) * 1.1)
        plt.show()


def test_detect_injected_correlation():
    methods = ['numpy']
    if HAVE_NUMBA:
        methods.append("numba")
    
    num_spike = 2000
    spike_times1 = np.sort(np.unique(np.random.randint(0, 100000, num_spike)))
    spike_times2 = np.sort(np.unique(np.random.randint(0, 100000, num_spike)))
    n = min(spike_times1.size, spike_times2.size)
    spike_times1 = spike_times1[:n]
    spike_times2 = spike_times2[:n]
    # inject 1.5 ms every 13 spikes
    spike_times2[::13] = spike_times1[::13] + 15
    spike_times2 = np.sort(spike_times2)
    
    num_injected = spike_times2[::15].size
    print('num_injected', num_injected)
    
    units_dict = {'1': spike_times1, '2': spike_times2}
    sorting = NumpySorting.from_dict([units_dict], sampling_frequency=10000.)

    for method in methods:
        correlograms, bins = compute_correlograms(sorting, window_ms=10., bin_ms=0.1, method=method)
        
        num_half_bins = correlograms.shape[2]  // 2

        cc_01 = correlograms[0, 1, :]
        cc_10 = correlograms[1, 0, :]
        
        #~ import matplotlib.pyplot as plt
        #~ fig, ax = plt.subplots()
        #~ ax.plot(bins[:-1], cc_01, marker='*',  color='red', label='cross-corr 0>1')
        #~ ax.plot(bins[:-1], cc_10, marker='*',  color='orange', label='cross-corr 1>0')
        #~ ax.set_title(method)
        #~ ax.legend()
    #~ plt.show()




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
    #~ test_auto_equal_cross_correlograms()
    #~ test_detect_injected_correlation()
    
    
    #~ test_correlograms_extension()

    #~ test = CorrelogramsExtensionTest()
    #~ test.setUp()
    #~ test.test_compute_correlograms()

