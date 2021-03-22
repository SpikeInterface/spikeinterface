'''
Comparison Widgets Gallery
===================================

Here is a gallery of all the available widgets using SortingExtractor objects.
'''

import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.widgets as sw

##############################################################################
# First, let's create a toy example with the `extractors` module:

recording, sorting_true = se.toy_example(duration=10, num_channels=4, seed=0, num_segments=1)

##############################################################################
# Let's run some spike sorting:

import spikeinterface.sorters as ss

sorting_SC = ss.run_spykingcircus(recording)
sorting_TDC = ss.run_tridesclous(recording)


##############################################################################
# Widgets using SortingComparison
# ---------------------------------
#
# We can compare the spike sorting output to the ground-truth sorting :code:`sorting_true` using the
# :code:`comparison` module. :code:`comp_SC` and :code:`comp_TDC` are :code:`SortingComparison` objects

import spikeinterface.comparison as sc

comp_SC = sc.compare_sorter_to_ground_truth(sorting_true, sorting_SC)
comp_TDC = sc.compare_sorter_to_ground_truth(sorting_true, sorting_TDC)

##############################################################################
# plot_confusion_matrix()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

w_comp_SC = sw.plot_confusion_matrix(comp_SC, count_text=True)
w_comp_TDC = sw.plot_confusion_matrix(comp_TDC, count_text=True)

##############################################################################
# plot_agreement_matrix()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

w_agr_SC = sw.plot_agreement_matrix(comp_SC, count_text=True)

##############################################################################
# plot_sorting_performance()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can also plot a performance metric (e.g. accuracy, recall, precision) with respect to a quality metric, for
# example signal-to-noise ratio. Quality metrics can be computed using the :code:`toolkit.validation` submodule

# TODO
# import spikeinterface.toolkit as st

# snrs = st.validation.compute_snrs(sorting_true, recording, save_as_property=True)

# w_perf = sw.plot_sorting_performance(comp_SC, property_name='snr', metric='accuracy')

##############################################################################
# Widgets using MultiSortingComparison
# -------------------------------------
#
# We can also compare all three SortingExtractor objects, obtaining a :code:`MultiSortingComparison` object.


multicomp = sc.compare_multiple_sorters([sorting_true, sorting_SC, sorting_TDC])

##############################################################################
# plot_multicomp_graph()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

w_multi = sw.plot_multicomp_graph(multicomp, edge_cmap='coolwarm', node_cmap='viridis', draw_labels=False,
                                  colorbar=True)


plt.show()