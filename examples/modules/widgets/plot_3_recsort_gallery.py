'''
Recording+Sorting Widgets Gallery
===================================================

Here is a gallery of all the available widgets using a pair of RecordingExtractor-SortingExtractor objects.
'''

import spikeinterface.extractors as se
import spikeinterface.widgets as sw

##############################################################################
# First, let's create a toy example with the `extractors` module:

recording, sorting = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)

##############################################################################
# plot_unit_waveforms()
# ~~~~~~~~~~~~~~~~~~~~~~~~

w_wf = sw.plot_unit_waveforms(recording, sorting, max_spikes_per_unit=100)

##############################################################################
# plot_amplitudes_distribution()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

w_ampd = sw.plot_amplitudes_distribution(recording, sorting, max_spikes_per_unit=300)

##############################################################################
# plot_amplitudes_timeseres()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

w_ampt = sw.plot_amplitudes_timeseries(recording, sorting, max_spikes_per_unit=300)

##############################################################################
# plot_pca_features()
# ~~~~~~~~~~~~~~~~~~~~~~~~

w_feat = sw.plot_pca_features(recording, sorting, colormap='rainbow', nproj=3, max_spikes_per_unit=100)
