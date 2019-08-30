'''
SortingExtractor Widgets Gallery
===================================

Here is a gallery of all the available widgets using SortingExtractor objects.
'''

import spikeinterface.extractors as se
import spikeinterface.widgets as sw

##############################################################################
# First, let's create a toy example with the `extractors` module:

recording, sorting = se.example_datasets.toy_example(duration=60, num_channels=4, seed=0)

##############################################################################
# plot_rasters()
# ~~~~~~~~~~~~~~~~~

w_rs = sw.plot_rasters(sorting)

##############################################################################
# plot_isi_distribution()
# ~~~~~~~~~~~~~~~~~~~~~~~~
w_isi = sw.plot_isi_distribution(sorting, bins=10, window=1)

##############################################################################
# plot_autocorrelograms()
# ~~~~~~~~~~~~~~~~~~~~~~~~
w_ach = sw.plot_autocorrelograms(sorting, bin_size=1, window=10, unit_ids=[1, 2, 4, 5, 8, 10, 7])

##############################################################################
# plot_crosscorrelograms()
# ~~~~~~~~~~~~~~~~~~~~~~~~
w_cch = sw.plot_crosscorrelograms(sorting, unit_ids=[1, 5, 8], bin_size=0.1, window=5)