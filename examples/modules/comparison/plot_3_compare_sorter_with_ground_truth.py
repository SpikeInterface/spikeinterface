"""
Compare spike sorting output with ground-truth recordings
=========================================================

Simulated recordings or paired pipette and extracellular recordings can
be used to validate spike sorting algorithms.

For comparing to ground-truth data, the
:py:func:`~spikeinterface.comparison.compare_sorter_to_ground_truth()` function
can be used. In this recording, we have ground-truth information for all
units, so we can set :code:`exhaustive_gt` to :code:`True`.

"""


##############################################################################
# Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

##############################################################################
# First, let's download a simulated dataset
#  from the repo 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'

local_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
recording, sorting_true = se.read_mearec(local_path)
print(recording)
print(sorting_true)


##############################################################################
# run herdingspikes on it

sorting_HS = ss.run_herdingspikes(recording)

##############################################################################

cmp_gt_HS = sc.compare_sorter_to_ground_truth(sorting_true, sorting_HS, exhaustive_gt=True)


##############################################################################
# To have an overview of the match we can use the unordered agreement matrix

sw.plot_agreement_matrix(cmp_gt_HS, ordered=False)

##############################################################################
# or ordered

sw.plot_agreement_matrix(cmp_gt_HS, ordered=True)

##############################################################################
# This function first matches the ground-truth and spike sorted units, and
# then it computes several performance metrics.
#
# Once the spike trains are matched, each spike is labeled as:
#
# - true positive (tp): spike found both in :code:`gt_sorting` and :code:`tested_sorting`
# - false negative (fn): spike found in :code:`gt_sorting`, but not in :code:`tested_sorting`
# - false positive (fp): spike found in :code:`tested_sorting`, but not in :code:`gt_sorting`
#
# From the counts of these labels the following performance measures are
# computed:
#
# -  accuracy: #tp / (#tp+ #fn + #fp)
# -  recall: #tp / (#tp + #fn)
# -  precision: #tp / (#tp + #fn)
# -  miss rate: #fn / (#tp + #fn1)
# -  false discovery rate: #fp / (#tp + #fp)
#
# The :code:`get_performance` method a pandas dataframe (or a dictionary if
# :code:`output='dict'`) with the comparison metrics. By default, these are
# calculated for each spike train of :code:`sorting1:code:`, the results can be
# pooled by average (average of the metrics) and by sum (all counts are
# summed and the metrics are computed then).

perf = cmp_gt_HS.get_performance()

##############################################################################
# Lets use seaborn swarm plot

fig1, ax1 = plt.subplots()
perf2 = pd.melt(perf, var_name='measurement')
ax1 = sns.swarmplot(data=perf2, x='measurement', y='value', ax=ax1)
ax1.set_xticklabels(labels=ax1.get_xticklabels(), rotation=45)

##############################################################################
# The confusion matrix is also a good summary of the score as it has
# the same shape as agreement matrix, but it contains an extra column for FN
# and an extra row for FP

sw.plot_confusion_matrix(cmp_gt_HS)

##############################################################################
# We can query the well and bad detected units. By default, the threshold
# on accuracy is 0.95.

cmp_gt_HS.get_well_detected_units()

##############################################################################

cmp_gt_HS.get_false_positive_units()

##############################################################################

cmp_gt_HS.get_redundant_units()



##############################################################################
# Lets do the same for tridesclous

sorting_TDC = ss.run_tridesclous(recording)
cmp_gt_TDC = sc.compare_sorter_to_ground_truth(sorting_true, sorting_TDC, exhaustive_gt=True)

##############################################################################

perf = cmp_gt_TDC.get_performance()

print(perf)

##############################################################################

sw.plot_agreement_matrix(cmp_gt_TDC, ordered=True)

##############################################################################
# Lets use seaborn swarm plot

fig2, ax2 = plt.subplots()
perf2 = pd.melt(perf, var_name='measurement')
ax2 = sns.swarmplot(data=perf2, x='measurement', y='value', ax=ax2)
ax2.set_xticklabels(labels=ax2.get_xticklabels(), rotation=45)

##############################################################################

print(cmp_gt_TDC.get_well_detected_units)

##############################################################################

print(cmp_gt_TDC.get_false_positive_units())

##############################################################################

print(cmp_gt_TDC.get_redundant_units())


plt.show()
