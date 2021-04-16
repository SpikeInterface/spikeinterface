'''
Recording+Sorting Widgets Gallery
===================================================

Here is a gallery of all the available widgets using a pair of RecordingExtractor-SortingExtractor objects.
'''
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.widgets as sw

##############################################################################
# First, let's download a simulated dataset
#  on the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording = se.MEArecRecordingExtractor(local_path)
sorting = se.MEArecSortingExtractor(local_path)
print(recording)
print(sorting)

##############################################################################
# Extract spike waveforms
# --------------------------
# 
# For convinience metris are computed on the WaveformExtractor object that gather recording/sorting and
# extracted waveforms in a single object

folder = 'waveforms_mearec'
we = si.extract_waveforms(recording, sorting, folder,
    load_if_exists=True,
    ms_before=1, ms_after=2., max_spikes_per_unit=500,
    n_jobs=1, chunk_size=30000)


##############################################################################
# plot_unit_waveforms()
# ~~~~~~~~~~~~~~~~~~~~~~~~

unit_ids = sorting.unit_ids[:4]

sw.plot_unit_waveforms(we, unit_ids=unit_ids)

##############################################################################
# plot_amplitudes_distribution()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO
# w_ampd = sw.plot_amplitudes_distribution(recording, sorting, max_spikes_per_unit=300)

##############################################################################
# plot_amplitudes_timeseres()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO
# w_ampt = sw.plot_amplitudes_timeseries(recording, sorting, max_spikes_per_unit=300)

##############################################################################
# plot_pca_features()
# ~~~~~~~~~~~~~~~~~~~~~~~~

# TODO
# w_feat = sw.plot_pca_features(recording, sorting, colormap='rainbow', nproj=3, max_spikes_per_unit=100)

plt.show()