'''
Widgets Gallery
===============

Here is a gallery of all the available widgets.

Let's firt import the required packages
'''

import spikeextractors as se
import spiketoolkit as st
import spikecomparison as sc
import spikewidgets as sw

##############################################################################
# First, let's create a toy example with the `extractors` module:

recording, sorting = se.example_datasets.toy_example(duration=60, num_channels=4, seed=0)

##############################################################################
# Widgets using RecordingExtractors
# ---------------------------------
#
# plot_timeseries()
# ~~~~~~~~~~~~~~~~~

w_ts = sw.plot_timeseries(recording)

w_ts1 = sw.plot_timeseries(recording, trange=[5, 15])

recording.set_channel_groups(channel_ids=recording.get_channel_ids(), groups=[0, 0, 1, 1])
w_ts2 = sw.plot_timeseries(recording, trange=[5, 15], color_groups=True)

##############################################################################
# **Note**: each function returns a widget object, which allows to access the figure and axis.

w_ts.figure.suptitle("Recording by group")
w_ts.ax.set_ylabel("Channel_ids")

##############################################################################
# plot_electrode_geometry()
# ~~~~~~~~~~~~~~~~~~~~~~~~~
w_el = sw.plot_electrode_geometry(recording)

##############################################################################
# Widgets using SortingExtractors
# ---------------------------------
#
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

##############################################################################
# Widgets using RecordingExtractors and SortingExtractors
# -------------------------------------------------------
#
# plot_unit_waveforms()
# ~~~~~~~~~~~~~~~~~~~~~~~~

w_wf = sw.plot_unit_waveforms(recording, sorting, max_num_waveforms=100)

##############################################################################
# plot_amplitudes_distribution()
# ~~~~~~~~~~~~~~~~~

w_ampd = sw.plot_amplitudes_distribution(recording, sorting, max_num_waveforms=300)

##############################################################################
# plot_autocorrelograms()
# ~~~~~~~~~~~~~~~~~

w_ampt = sw.plot_amplitudes_timeseres(recording, sorting, max_num_waveforms=300)

##############################################################################
# plot_features()
# ~~~~~~~~~~~~~~~~~~~~~~~~

w_feat = sw.plot_features(recording, sorting, colormap='rainbow', nproj=3, max_num_waveforms=100)

##############################################################################
# Widgets using SortingComparison
# ---------------------------------
#
# We will compare the same :code:`sorting` object to show the widgets for the :code:`SortingComparison` class.

comp = sc.compare_sorter_to_ground_truth(sorting, sorting)

##############################################################################
# plot_confusion_matrix()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

w_comp = sw.plot_confusion_matrix(comp, count_text=False)


##############################################################################
# Widgets using MultiSortingComparison
# ---------------------------------
#
# We will compare the same :code:`sorting` object 3 times to show the widgets for the :code:`MultiSortingComparison`
# class.

multicomp = sc.compare_multiple_sorters([sorting, sorting, sorting])

##############################################################################
# plot_multicomp_graph()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

w_multi = sw.plot_multicomp_graph(multicomp, edge_cmap='coolwarm', node_cmap='viridis', draw_labels=False,
                                  colorbar=True)