"""
Compare multiple sorters and consensus based method
====================================================

With 3 or more spike sorters, the comparison is implemented with a
graph-based method. The multiple sorter comparison also allows to clean
the output by applying a consensus-based method which only selects spike
trains and spikes in agreement with multiple sorters.

"""


##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.sorters as sorters
import spikeinterface.comparison as sc

##############################################################################
# First, let's create a toy example:

recording, sorting = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)


#############################################################################
# Then run 3 spike sorters and compare their ouput.

sorting_KL = sorters.run_klusta(recording)
sorting_MS4 = sorters.run_mountainsort4(recording)
sorting_TDC = sorters.run_tridesclous(recording)

#############################################################################
# Compare multiple spike sorter outputs
# -------------------------------------------

mcmp = sc.compare_multiple_sorters(sorting_list=[sorting_KL, sorting_MS4, sorting_TDC], 
                                              name_list=['KL', 'MS4', 'TDC'], verbose=True)


#############################################################################
# The multiple sorters comparison internally computes pairwise comparison,
# that can be accessed as follows:

mcmp.sorting_comparisons['KL']['TDC'].get_mapped_sorting1().get_mapped_unit_ids()

#############################################################################

mcmp.sorting_comparisons['MS4']['TDC'].get_mapped_sorting1().get_mapped_unit_ids()

#############################################################################
# We can see that there is a better agreement between tridesclous and
# mountainsort (5 units matched), while klusta only has two matched units
# with tridesclous, and three with mountainsort.


#############################################################################
# Consensus-based method
# ---------------------------
# 
# We can pull the units in agreement with different sorters using the
# :code:`get_agreement_sorting` method. This allows to make spike sorting more
# robust by integrating the output of several algorithms. On the other
# hand, it might suffer from weak performance of single algorithms.
# 
# When extracting the units in agreement, the spike trains are modified so
# that only the true positive spikes between the comparison with the best
# match are used.

agr_3 = mcmp.get_agreement_sorting(minimum_matching=3)
print('Units in agreement for all three sorters: ', agr_3.get_unit_ids())

#############################################################################

agr_2 = mcmp.get_agreement_sorting(minimum_matching=2)
print('Units in agreement for at least two sorters: ', agr_2.get_unit_ids())

#############################################################################

agr_all = mcmp.get_agreement_sorting()

#############################################################################
# The unit index of the different sorters can also be retrieved from the
# agreement sorting object (:code:`agr_3`) property :code:`sorter_unit_ids`.

print(agr_3.get_unit_property_names())

#############################################################################

print(agr_3.get_unit_ids())
# take one unit in agreement
u = agr_3.get_unit_ids()[1]
print(agr_3.get_unit_property(u, 'sorter_unit_ids'))

#############################################################################
# Now that we found our unit, we can plot a rasters with the spike trains
# of the single sorters and the one from the consensus based method. When
# extracting the agreement sorting, spike trains are cleaned so that only
# true positives remain from the comparison with the largest agreement are
# kept. Let’s take a look at the raster plots for the different sorters
# and the agreement sorter:

d = agr_3.get_unit_property(u, 'sorter_unit_ids')
st0 =sorting_KL.get_unit_spike_train(d['KL'])
st1 = sorting_MS4.get_unit_spike_train(d['MS4'])
st2 = sorting_TDC.get_unit_spike_train(d['TDC'])
st3 = agr_3.get_unit_spike_train(u)

fig, ax = plt.subplots()
ax.plot(st0, 0 * np.ones(st0.size), '|')
ax.plot(st1, 1 * np.ones(st1.size), '|')
ax.plot(st2, 2 * np.ones(st2.size), '|')
ax.plot(st3, 3 * np.ones(st3.size), '|')

print('Klusta spike train length', st0.size)
print('Mountainsort spike train length', st1.size)
print('Tridesclous spike train length', st2.size)
print('Agreement spike train length', st3.size)

#############################################################################
# As we can see, the best match is between Mountainsort and Tridesclous,
# but only the true positive spikes make up the agreement spike train.

plt.show()






