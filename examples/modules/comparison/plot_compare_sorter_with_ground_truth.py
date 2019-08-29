"""
Compare spike sprting output with ground-truth recordings
==================================

Simulated recordings or paired pipette and extracellular recordings can
be used to validate spike sorting algorithms.

For comparing to ground-truth data, the
``compare_sorter_to_ground_truth(gt_sorting, tested_sorting)`` function
can be used. In this recording, we have ground-truth information for all
units, so we can set ``exhaustive_gt`` to ``True``.

"""


##############################################################################
# Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO fix import
import spikeextractors as se
import spikeinterface.sorters as sorters
import spikeinterface.comparison as sc

##############################################################################

recording, sorting_true = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)

sorting_MS4 = sorters.run_mountainsort4(recording)

##############################################################################

cmp_gt_MS4 = sc.compare_sorter_to_ground_truth(sorting_true, sorting_MS4, exhaustive_gt=True)

##############################################################################
# This function first matches the ground-truth and spike sorted units, and
# then it computes several performance metrics.
# 
# Once the spike trains are matched, each spike is labelled as: - true
# positive (tp): spike found both in ``gt_sorting`` and ``tested_sorting``
# - false negative (fn): spike found in ``gt_sorting``, but not in
# ``tested_sorting`` - false positive (fp): spike found in
# ``tested_sorting``, but not in ``gt_sorting`` - misclassification errors
# (cl): spike found in ``gt_sorting``, not in ``tested_sorting``, found in
# another matched spike train of ``tested_sorting``, and not labelled as
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
# The ``get_performance`` method a pandas dataframe (or a dictionary if
# ``output='dict'``) with the comparison metrics. By default, these are
# calculated for each spike train of ``sorting1``, the results can be
# pooles by average (average of the metrics) and by sum (all counts are
# summed and the metrics are computed then).

perf = cmp_gt_MS4.get_performance()

##############################################################################
# Lets use seaborn swarm plot

fig, ax = plt.subplots()
perf2 = pd.melt(perf, var_name='measurement')
sns.swarmplot(data=perf2, x='measurement', y='value')


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

fig, ax = plt.subplots()
perf2 = pd.melt(perf, var_name='measurement')
sns.swarmplot(data=perf2, x='measurement', y='value')

##############################################################################

cmp_gt_KL.get_well_detected_units

##############################################################################

cmp_gt_KL.get_false_positive_units()

##############################################################################

cmp_gt_KL.get_redundant_units()


