'''
Peaks Widgets Gallery
=====================

Some widgets are useful before sorting and works with "peaks" given by detect_peaks()
function.

They are useful to check drift before running sorters.

'''
import matplotlib.pyplot as plt

import spikeinterface.full as si

##############################################################################
# First, let's download a simulated dataset
# from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
rec, sorting = si.read_mearec(local_path)


##############################################################################
# Lets filter and detect peak on it

from spikeinterface.sortingcomponents.peak_detection import detect_peaks

rec_filtred = si.bandpass_filter(rec, freq_min=300., freq_max=6000., margin_ms=5.0)
print(rec_filtred)
peaks = detect_peaks(
        rec_filtred, method='locally_exclusive',
        peak_sign='neg', detect_threshold=6, exclude_sweep_ms=0.3,
        local_radius_um=100,
        noise_levels=None,
        random_chunk_kwargs={},
        chunk_memory='10M', n_jobs=1, progress_bar=True)

##############################################################################
# peaks is a numpy 1D array with structured dtype that contains several fields:

print(peaks.dtype)
print(peaks.shape)
print(peaks.dtype.fields.keys())

##############################################################################
# This "peaks" vector can be used in several widgets, for instance
# plot_peak_activity_map()

si.plot_peak_activity_map(rec_filtred, peaks=peaks)

##############################################################################
# can be also animated with bin_duration_s=1.

si.plot_peak_activity_map(rec_filtred, bin_duration_s=1.)


plt.show()
