"""
Peaks Widgets Gallery
=====================

Some widgets are useful before sorting and works with the "peaks" given by the detect_peaks()
function.

They are useful to check drift before running sorters.

"""

import matplotlib.pyplot as plt

import spikeinterface.full as si

##############################################################################
# First, let's download a simulated dataset
# from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
rec, sorting = si.read_mearec(local_path)


##############################################################################
# Let's filter and detect peaks on it

from spikeinterface.sortingcomponents.peak_detection import detect_peaks

rec_filtred = si.bandpass_filter(recording=rec, freq_min=300.0, freq_max=6000.0, margin_ms=5.0)
print(rec_filtred)
peaks = detect_peaks(
    recording=rec_filtred,
    method="locally_exclusive",
    peak_sign="neg",
    detect_threshold=6,
    exclude_sweep_ms=0.3,
    radius_um=100,
    noise_levels=None,
    random_chunk_kwargs={},
    chunk_memory="10M",
    n_jobs=1,
    progress_bar=True,
)

##############################################################################
# peaks is a numpy 1D array with structured dtype that contains several fields:

print(peaks.dtype)
print(peaks.shape)
print(peaks.dtype.fields.keys())

##############################################################################
# This "peaks" vector can be used in several widgets, for instance
# plot_peak_activity()

si.plot_peak_activity(recording=rec_filtred, peaks=peaks)

plt.show()

##############################################################################
# can be also animated with bin_duration_s=1.

si.plot_peak_activity(recording=rec_filtred, peaks=peaks, bin_duration_s=1.0)


plt.show()
