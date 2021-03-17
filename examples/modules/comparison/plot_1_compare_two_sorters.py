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
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

##############################################################################
# First, let's create a toy example:

recording, sorting = se.toy_example(num_channels=4, duration=10, seed=0, num_segments=1)


#############################################################################
# Then run two spike sorters and compare their ouput.

sorting_SC = ss.run_spykingcircus(recording)
sorting_TDC = ss.run_tridesclous(recording)


#############################################################################
# The :code:`compare_two_sorters` function allows us to compare the spike
# sorting output. It returns a :code:`SortingComparison` object, with methods
# to inspect the comparison output easily. The comparison matches the
# units by comparing the agreement between unit spike trains.
# 
# Let’s see how to inspect and access this matching.

cmp_SC_TDC = sc.compare_two_sorters(sorting1=sorting_SC, sorting2=sorting_TDC, 
                                               sorting1_name='SC', sorting2_name='TDC')

#############################################################################
# We can check the agreement matrix to inspect the matching.

sw.plot_agreement_matrix(cmp_SC_TDC)

#############################################################################
# Some useful internal dataframes help to check the match and count
#  like **match_event_count** or **agreement_scores**

print(cmp_SC_TDC.match_event_count)
print(cmp_SC_TDC.agreement_scores)

#############################################################################
# In order to check which units were matched, the :code:`get_matching`
# methods can be used. If units are not matched they are listed as -1.

sc_to_tdc, tdc_to_sc = cmp_SC_TDC.get_matching()

print('matching SC to TDC')
print(sc_to_tdc)
print('matching TDC to SC')
print(tdc_to_sc)


#############################################################################
# The :code:get_unit_spike_train` returns the mapped spike train. We can use
# it to check the spike times.

matched_ids = sc_to_tdc[sc_to_tdc != -1]

unit_id_SC = matched_ids.index[0]
unit_id_TDC = matched_ids[unit_id_SC]



# check that matched spike trains correspond
st1 = sorting_SC.get_unit_spike_train(unit_id_SC)
st2 = sorting_TDC.get_unit_spike_train(unit_id_TDC)
fig, ax = plt.subplots()
ax.plot(st1, np.zeros(st1.size), '|')
ax.plot(st2, np.ones(st2.size), '|')

plt.show()
