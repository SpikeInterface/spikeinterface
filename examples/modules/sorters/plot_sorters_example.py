"""
Run spike sorting algorithms
============================

This example shows the basic usage of the :code:`sorters` module of :code:`spikeinterface`
"""

import spikeextractors as se
import spikeinterface.sorters as ss

##############################################################################
# First, let's create a toy example:

recording, sorting_true = se.example_datasets.toy_example(duration=60, seed=0)

##############################################################################
# Check available sorters
# --------------------------
# 

print(ss.available_sorters())

##############################################################################
# This will list the sorters installed in the machine. Each spike sorter
# is implemented in a class. To access the class names you can run:

print(ss.installed_sorter_list)

##############################################################################
# 2) Check and set sorters parameters
# -----------------------------------
# 

default_ms4_params = ss.Mountainsort4Sorter.default_params()
print(default_ms4_params)

##############################################################################
#  Parameters can be changed either by passing a full dictionary, or by
# passing single arguments.

# Mountainsort4 spike sorting
default_ms4_params['detect_threshold'] = 4
default_ms4_params['curation'] = False

# parameters set by params dictionary
sorting_MS4 = ss.run_mountainsort4(recording=recording, **default_ms4_params,
                                   output_folder='tmp_MS4')
##############################################################################

# parameters set by params dictionary
sorting_MS4_10 = ss.run_mountainsort4(recording=recording, detect_threshold=10,
                                      output_folder='tmp_MS4')

##############################################################################

print('Units found with threshold = 4:', sorting_MS4.get_unit_ids())
print('Units found with threshold = 10:', sorting_MS4_10.get_unit_ids())

##############################################################################
# 3) Run sorters
# --------------

# SpyKING Circus spike sorting
# sorting_SC = ss.run_spykingcircus(recording, output_folder='tmp_SC')
# print('Units found with Spyking Circus:', sorting_SC.get_unit_ids())

##############################################################################

# KiloSort spike sorting (KILOSORT_PATH and NPY_MATLAB_PATH can be set as environment variables)
# sorting_KS = ss.run_kilosort(recording, output_folder='tmp_KS')
#  print('Units found with Kilosort:', sorting_KS.get_unit_ids())

##############################################################################

# Kilosort2 spike sorting (KILOSORT2_PATH and NPY_MATLAB_PATH can be set as environment variables)
# sorting_KS2 = ss.run_kilosort2(recording, output_folder='tmp_KS2')
#  print('Units found with Kilosort2', sorting_KS2.get_unit_ids())

##############################################################################

# Klusta spike sorting
#  sorting_KL = ss.run_klusta(recording, output_folder='tmp_KL')
# print('Units found with Klusta:', sorting_KL.get_unit_ids())

##############################################################################

# IronClust spike sorting (IRONCLUST_PATH can be set as environment variables)
# sorting_IC = ss.run_ironclust(recording, output_folder='tmp_IC')
# print('Units found with Ironclust:', sorting_IC.get_unit_ids())

##############################################################################

# Tridesclous spike sorting
# sorting_TDC = ss.run_tridesclous(recording, output_folder='tmp_TDC')
# print('Units found with Tridesclous:', sorting_TDC.get_unit_ids())
