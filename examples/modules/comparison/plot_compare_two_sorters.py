"""
Compare two sorters
====================

This example show how to compare the result of two sorters.

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
# Then run two spike sorters and compare their ouput.

sorting_KL = sorters.run_klusta(recording)
sorting_MS4 = sorters.run_mountainsort4(recording)

#############################################################################
# The :code:`compare_two_sorters` function allows us to compare the spike
# sorting output. It returns a :code:`SortingComparison` object, with methods
# to inspect the comparison output easily. The comparison matches the
# units by comparing the agreement between unit spike trains.
# 
# Let’s see how to inspect and access this matching.

cmp_KL_MS4 = sc.compare_two_sorters(sorting1=sorting_KL, sorting2=sorting_MS4, 
                                               sorting1_name='klusta', sorting2_name='ms4')

#############################################################################
# In order to check which units were matched, the :code:`get_mapped_sorting`
# methods can be used. If units are not matched they are listed as -1.

# units matched to klusta units
mapped_sorting_klusta = cmp_KL_MS4.get_mapped_sorting1()
print('Klusta units:', sorting_KL.get_unit_ids())
print('Klusta mapped units:', mapped_sorting_klusta.get_mapped_unit_ids())

# units matched to ms4 units
mapped_sorting_ms4 = cmp_KL_MS4.get_mapped_sorting2()
print('Mountainsort units:',sorting_MS4.get_unit_ids())
print('Mountainsort mapped units:',mapped_sorting_ms4.get_mapped_unit_ids())

#############################################################################
# The :code:get_unit_spike_train` returns the mapped spike train. We can use
# it to check the spike times.

# find a unit from KL that have a match
ind  = np.where(np.array(mapped_sorting_klusta.get_mapped_unit_ids())!=-1)[0][0]
u1 = sorting_KL.get_unit_ids()[ind]
print(ind, u1)

# check that matched spike trains correspond
st1 = sorting_KL.get_unit_spike_train(u1)
st2 = mapped_sorting_klusta.get_unit_spike_train(u1)
fig, ax = plt.subplots()
ax.plot(st1, np.zeros(st1.size), '|')
ax.plot(st2, np.ones(st2.size), '|')

plt.show()
