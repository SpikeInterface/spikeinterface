'''
Comparison Widgets Gallery
===================================

Here is a gallery of all the available widgets using SortingExtractor objects.
'''

import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

##############################################################################
# First, let's create a toy example with the `extractors` module:

recording, sorting_true = se.toy_example(duration=10, num_channels=4, seed=0, num_segments=1)
recording = recording.save()

##############################################################################
# Let's run some spike sorting:


sorting_SC = ss.run_spykingcircus(recording)
sorting_TDC = ss.run_tridesclous(recording)


##############################################################################
# Widgets using SortingComparison
# ---------------------------------
#
# We can compare the spike sorting output to the ground-truth sorting :code:`sorting_true` using the
# :code:`comparison` module. :code:`comp_SC` and :code:`comp_TDC` are :code:`SortingComparison` objects

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

we = si.extract_waveforms(recording, sorting_true, 'waveforms_mearec',
    load_if_exists=True,
    ms_before=1, ms_after=2., max_spikes_per_unit=500,
    n_jobs=1, chunk_size=30000)


metrics = st.compute_quality_metrics(we, metric_names=['snr'])

sw.plot_sorting_performance(comp_SC, metrics, performance_name='accuracy', metric_name='snr')
sw.plot_sorting_performance(comp_TDC, metrics, performance_name='accuracy', metric_name='snr')

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