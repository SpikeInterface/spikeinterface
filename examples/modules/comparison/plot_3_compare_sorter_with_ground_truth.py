"""
Compare spike sorting output with ground-truth recordings
=========================================================

Simulated recordings or paired pipette and extracellular recordings can
be used to validate spike sorting algorithms.

For comparing to ground-truth data, the
:code:`compare_sorter_to_ground_truth(gt_sorting, tested_sorting)` function
can be used. In this recording, we have ground-truth information for all
units, so we can set :code:`exhaustive_gt` to :code:`True`.

"""


##############################################################################
# Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import spikeinterface.extractors as se
import spikeinterface.sorters as sorters
import spikeinterface.comparison as sc

##############################################################################

recording, sorting_true = se.example_datasets.toy_example(num_channels=4, duration=10, seed=0)

sorting_MS4 = sorters.run_mountainsort4(recording)

##############################################################################

cmp_gt_MS4 = sc.compare_sorter_to_ground_truth(sorting_true, sorting_MS4, exhaustive_gt=True)

##############################################################################
# This function first matches the ground-truth and spike sorted units, and
# then it computes several performance metrics.
# 
# Once the spike trains are matched, each spike is labelled as: - true
# positive (tp): spike found both in :code:`gt_sorting` and :code:`tested_sorting`
# - false negative (fn): spike found in :code:`gt_sorting`, but not in
# :code:`tested_sorting` - false positive (fp): spike found in
# :code:`tested_sorting`, but not in :code:`gt_sorting` - misclassification errors
# (cl): spike found in :code:`gt_sorting`, not in :code:`tested_sorting`, found in
# another matched spike train of :code:`tested_sorting`, and not labelled as
# true positives
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
# pooles by average (average of the metrics) and by sum (all counts are
# summed and the metrics are computed then).

perf = cmp_gt_MS4.get_performance()

##############################################################################
# Lets use seaborn swarm plot

fig1, ax1 = plt.subplots()
perf2 = pd.melt(perf, var_name='measurement')
ax1 = sns.swarmplot(data=perf2, x='measurement', y='value', ax=ax1)
ax1.set_xticklabels(labels=ax1.get_xticklabels(), rotation=45)


##############################################################################
# We can query the well and bad detected units. By default, the threshold
# on accuracy is 0.95.

cmp_gt_MS4.get_well_detected_units()

##############################################################################

cmp_gt_MS4.get_false_positive_units()

##############################################################################

cmp_gt_MS4.get_redundant_units()



##############################################################################
# Lets do the same for klusta

sorting_KL = sorters.run_klusta(recording)
cmp_gt_KL = sc.compare_sorter_to_ground_truth(sorting_true, sorting_KL, exhaustive_gt=True)

##############################################################################

perf = cmp_gt_KL.get_performance()

##############################################################################
# Lets use seaborn swarm plot

fig2, ax2 = plt.subplots()
perf2 = pd.melt(perf, var_name='measurement')
ax2 = sns.swarmplot(data=perf2, x='measurement', y='value', ax=ax2)
ax2.set_xticklabels(labels=ax2.get_xticklabels(), rotation=45)

##############################################################################

print(cmp_gt_KL.get_well_detected_units)

##############################################################################

print(cmp_gt_KL.get_false_positive_units())

##############################################################################

print(cmp_gt_KL.get_redundant_units())


