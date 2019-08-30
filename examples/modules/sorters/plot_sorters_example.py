"""
Run sorters
===========

This notebook shows how to use the spikeinterface.sorters module to:


1. check available sorters
2. check and set sorters parameters
3. run sorters
4. use the spike sorter launcher
5. spike sort by property

"""

# TODO fix import
import spikeextractors as se
import spikeinterface.sorters as sorters
import time

##############################################################################
# First, let's create a toy example:

recording, sorting_true = se.example_datasets.toy_example(duration=60, seed=0)



##############################################################################
# 1) Check available sorters
# --------------------------
# 

import spikeinterface.sorters as sorters

print(sorters.available_sorters())


##############################################################################
# This will list the sorters installed in the machine. Each spike sorter
# is implemented in a class. To access the class names you can run:

print(sorters.installed_sorter_list)


##############################################################################
# 2) Check and set sorters parameters
# -----------------------------------
# 

default_ms4_params = sorters.Mountainsort4Sorter.default_params()
print(default_ms4_params)

##############################################################################
# Parameters can be changed either by passing a full dictionary, or by
# passing single arguments.

# Mountainsort4 spike sorting
default_ms4_params['detect_threshold'] = 4
default_ms4_params['curation'] = False

# parameters set by params dictionary
sorting_MS4 = sorters.run_mountainsort4(recording=recording, **default_ms4_params, 
                                            output_folder='tmp_MS4')
##############################################################################

# parameters set by params dictionary
sorting_MS4_10 = sorters.run_mountainsort4(recording=recording, detect_threshold=10, 
                                            output_folder='tmp_MS4')

##############################################################################

print('Units found with threshold = 4:', sorting_MS4.get_unit_ids())
print('Units found with threshold = 10:', sorting_MS4_10.get_unit_ids())


##############################################################################
# 3) Run sorters
# --------------

# SpyKING Circus spike sorting
sorting_SC = sorters.run_spykingcircus(recording, output_folder='tmp_SC')
print('Units found with Spyking Circus:', sorting_SC.get_unit_ids())

##############################################################################

# KiloSort spike sorting (KILOSORT_PATH and NPY_MATLAB_PATH can be set as environment variables)
# sorting_KS = sorters.run_kilosort(recording, output_folder='tmp_KS')
# print('Units found with Kilosort:', sorting_KS.get_unit_ids())

##############################################################################

# Kilosort2 spike sorting (KILOSORT2_PATH and NPY_MATLAB_PATH can be set as environment variables)
# sorting_KS2 = sorters.run_kilosort2(recording, output_folder='tmp_KS2')
# print('Units found with Kilosort2', sorting_KS2.get_unit_ids())

##############################################################################

# Klusta spike sorting
# sorting_KL = sorters.run_klusta(recording, output_folder='tmp_KL')
# print('Units found with Klusta:', sorting_KL.get_unit_ids())

##############################################################################

# IronClust spike sorting (IRONCLUST_PATH can be set as environment variables)
# sorting_IC = sorters.run_ironclust(recording, output_folder='tmp_IC')
# print('Units found with Ironclust:', sorting_IC.get_unit_ids())

##############################################################################

# Tridesclous spike sorting
sorting_TDC = sorters.run_tridesclous(recording, output_folder='tmp_TDC')
print('Units found with Tridesclous:', sorting_TDC.get_unit_ids())


##############################################################################
# 4) Use the spike sorter launcher
# --------------------------------
# The launcher enables to call any spike sorter with the same functions:
# ``run_sorter`` and ``run_sorters``. For running multiple sorters on the
# same recording extractor or a collection of them, the ``run_sorters``
# function can be used.

recording_list = [recording]
sorter_list = ['klusta', 'mountainsort4', 'tridesclous']
sorting_output = sorters.run_sorters(sorter_list, recording_list, working_folder='tmp_some_sorters')

##############################################################################
# retrieve results for all (recording, sorter) pairs
# access with dict with tupple of keys


for (rec, sorter), extractor in sorting_output.items():
    print(rec, sorter, ':', extractor.get_unit_ids())

##############################################################################
# 5) Spike sort by property
# -------------------------
# Sometimes, you might want to sort your data depending on a specific
# property of your recording channels.
# 
# For example, when using multiple tetrodes, a good idea is to sort each
# tetrode separately. In this case, channels belonging to the same tetrode
# will be in the same 'group'. Alternatively, for long silicon probes,
# such as Neuropixels, you could sort different areas separately, for
# example hippocampus and thalamus.
# 
# All this can be done by sorting by 'property'. Properties can be loaded
# to the recording channels either manually (using the
# ``set_channel_property`` method, or by using a probe file. In this
# example we will create a 16 channel recording and split it in four
# tetrodes.


recording_tetrodes, sorting_true = se.example_datasets.toy_example(duration=60, num_channels=16)

# initially there is no group information
print(recording_tetrodes.get_shared_channel_property_names())

##############################################################################
# The file tetrode_16.prb contain the channel group description
# 
# .. parsed-literal::
# 
#     channel_groups = {
#         0: {
#             'channels': [0,1,2,3],
#         },
#         1: {
#             'channels': [4,5,6,7],
#         },
#         2: {
#             'channels': [8,9,10,11],
#         },
#         3: {
#             'channels': [12,13,14,15],
#         }
#     }

# load probe file to add group information
recording_tetrodes = se.load_probe_file(recording_tetrodes, 'tetrode_16.prb')
print(recording_tetrodes.get_shared_channel_property_names())

##############################################################################
# We can now use the launcher to spike sort by the property 'group'. The
# different groups can also be sorted in parallel, and the output sorting
# extractor will have the same property used for sorting. Running in
# parallel (in thread) can speed up the computations.

t_start = time.time()
sorting_tetrodes = sorters.run_sorter('klusta', recording_tetrodes, output_folder='tmp_tetrodes', 
                                         grouping_property='group', parallel=False)
print('Elapsed time: ', time.time() - t_start)

##############################################################################


t_start = time.time()
sorting_tetrodes_p = sorters.run_sorter('klusta', recording_tetrodes, output_folder='tmp_tetrodes', 
                                           grouping_property='group', parallel=True)
print('Elapsed time parallel: ', time.time() - t_start)


##############################################################################

print('Units non parallel: ', sorting_tetrodes.get_unit_ids())
print('Units parallel: ', sorting_tetrodes_p.get_unit_ids())

##############################################################################

# Now that spike sorting is done, it's time to do some postprocessing,
# comparison, and validation of the results!


