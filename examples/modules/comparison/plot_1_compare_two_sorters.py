"""
Compare two sorters
====================

This example show how to compare the result of two sorters.

"""


##############################################################################
# Import

import numpy as np
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

##############################################################################
# First, let's download a simulated dataset
#  from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording, sorting = se.read_mearec(local_path)
print(recording)
print(sorting)



#############################################################################
# Then run two spike sorters and compare their output.

sorting_HS = ss.run_herdingspikes(recording)
sorting_TDC = ss.run_tridesclous(recording)


#############################################################################
# The :py:func:`~spikeinterface.comparison.compare_two_sorters` function allows us to compare the spike
# sorting output. It returns a :py:class:`~spikeinterface.comparison.SymmetricSortingComparison` object, with methods
# to inspect the comparison output easily. The comparison matches the
# units by comparing the agreement between unit spike trains.
#
# Let’s see how to inspect and access this matching.

cmp_HS_TDC = sc.compare_two_sorters(
    sorting1=sorting_HS,
    sorting2=sorting_TDC,
    sorting1_name='HS',
    sorting2_name='TDC',
)

#############################################################################
# We can check the agreement matrix to inspect the matching.

sw.plot_agreement_matrix(cmp_HS_TDC)

#############################################################################
# Some useful internal dataframes help to check the match and count
#  like **match_event_count** or **agreement_scores**

print(cmp_HS_TDC.match_event_count)
print(cmp_HS_TDC.agreement_scores)

#############################################################################
# In order to check which units were matched, the :code:`get_matching`
# methods can be used. If units are not matched they are listed as -1.

sc_to_tdc, tdc_to_sc = cmp_HS_TDC.get_matching()

print('matching HS to TDC')
print(sc_to_tdc)
print('matching TDC to HS')
print(tdc_to_sc)


#############################################################################
# The :py:func:`~spikeinterface.core.BaseSortingSegment.get_unit_spike_train` returns the mapped spike train. We
# can use
# it to check the spike times.

matched_ids = sc_to_tdc[sc_to_tdc != -1]

unit_id_HS = matched_ids.index[0]
unit_id_TDC = matched_ids[unit_id_HS]



# check that matched spike trains correspond
st1 = sorting_HS.get_unit_spike_train(unit_id_HS)
st2 = sorting_TDC.get_unit_spike_train(unit_id_TDC)
fig, ax = plt.subplots()
ax.plot(st1, np.zeros(st1.size), '|')
ax.plot(st2, np.ones(st2.size), '|')

plt.show()
