"""
Ground truth study tutorial
==================================

This tutorial illustrates how to run a "study".
A study is a systematic performance comparisons several ground truth
recordings with several sorters.

The submodule study and the class  propose high level tools functions
to run many groundtruth comparison with many sorter on many recordings
and then collect and aggregate results in an easy way.

The all mechanism is based on an intrinsic organization
into a "study_folder" with several subfolder:

* raw_files : contain a copy in binary format of recordings
* sorter_folders : contains output of sorters
* ground_truth : contains a copy of sorting ground  in npz format
* sortings: contains light copy of all sorting in npz format
* tables: some table in cvs format

In order to run and rerun the computation all gt_sorting and
recordings are copied to a fast and universal format :
binary (for recordings) and npz (for sortings).
"""

##############################################################################
# Imports

import matplotlib.pyplot as plt
import seaborn as sns

import spikeinterface.extractors as se
import spikeinterface.widgets as sw
from spikeinterface.comparison import GroundTruthStudy

##############################################################################
# Setup study folder and run all sorters
# --------------------------------------
#
# We first generate the folder.
# this can take some time because recordings are copied inside the folder.


rec0, gt_sorting0 = se.toy_example(num_channels=4, duration=10, seed=10, num_segments=1)
rec1, gt_sorting1 = se.toy_example(num_channels=4, duration=10, seed=0, num_segments=1)
gt_dict = {
    'rec0': (rec0, gt_sorting0),
    'rec1': (rec1, gt_sorting1),
}
study_folder = 'a_study_folder'
study = GroundTruthStudy.create(study_folder, gt_dict)

##############################################################################
# Then just run all sorters on all recordings in one functions.

# sorter_list = st.sorters.available_sorters() # this get all sorters.
sorter_list = ['herdingspikes', 'tridesclous', ]
study.run_sorters(sorter_list, mode_if_folder_exists="keep")

##############################################################################
# You can re run **run_study_sorters** as many time as you want.
# By default **mode='keep'** so only uncomputed sorters are rerun.
# For instance, so just remove the "sorter_folders/rec1/herdingspikes" to re-run
# only one sorter on one recording.
#
# Then we copy the spike sorting outputs into a separate subfolder.
# This allow to remove the "large" sorter_folders.

study.copy_sortings()

##############################################################################
# Collect comparisons
# -------------------
#  
# You can collect in one shot all results and run the
# GroundTruthComparison on it.
# So you can access finely to all individual results.
#  
# Note that exhaustive_gt=True when you exactly how many
# units in ground truth (for synthetic datasets)

study.run_comparisons(exhaustive_gt=True)

for (rec_name, sorter_name), comp in study.comparisons.items():
    print('*' * 10)
    print(rec_name, sorter_name)
    print(comp.count_score)  # raw counting of tp/fp/...
    comp.print_summary()
    perf_unit = comp.get_performance(method='by_unit')
    perf_avg = comp.get_performance(method='pooled_with_average')
    m = comp.get_confusion_matrix()
    w_comp = sw.plot_agreement_matrix(comp)
    w_comp.ax.set_title(rec_name  + ' - ' + sorter_name)

##############################################################################
# Collect synthetic dataframes and display
# ----------------------------------------
#
# As shown previously, the performance is returned as a pandas dataframe.
# The :py:func:`~spikeinterface.comparison.aggregate_performances_table()` function, gathers all the outputs in
# the study folder and merges them in a single dataframe.

dataframes = study.aggregate_dataframes()

##############################################################################
# Pandas dataframes can be nicely displayed as tables in the notebook.

print(dataframes.keys())

##############################################################################

print(dataframes['run_times'])

##############################################################################
# Easy plot with seaborn
# ----------------------
#  
# Seaborn allows to easily plot pandas dataframes. Let’s see some
# examples.

run_times = dataframes['run_times']
fig1, ax1 = plt.subplots()
sns.barplot(data=run_times, x='rec_name', y='run_time', hue='sorter_name', ax=ax1)
ax1.set_title('Run times')

##############################################################################

perfs = dataframes['perf_by_unit']
fig2, ax2 = plt.subplots()
sns.swarmplot(data=perfs, x='sorter_name', y='recall', hue='rec_name', ax=ax2)
ax2.set_title('Recall')
ax2.set_ylim(-0.1, 1.1)

plt.show()