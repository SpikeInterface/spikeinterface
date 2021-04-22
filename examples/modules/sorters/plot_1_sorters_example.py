"""
Run spike sorting algorithms
============================

This example shows the basic usage of the :code:`sorters` module of :code:`spikeinterface`
"""

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from pprint import pprint

##############################################################################
# First, let's create a toy example:
# We choose explicitly one segment because many sorters handle only recording with unique segment

recording, sorting_true = se.toy_example(duration=10, seed=0, num_segments=1)
print(recording)
print(sorting_true)

##############################################################################
# Check available sorters
# --------------------------
# 

pprint(ss.available_sorters())

##############################################################################
# This will list the sorters available through SpikeInterface. To see which sorters are installed on the machine
# you can run:

pprint(ss.installed_sorters())

##############################################################################
# Change sorter parameters
# -----------------------------------
# 

default_SC_params = ss.SpykingcircusSorter.default_params()
pprint(default_SC_params)

##############################################################################
#  Parameters can be changed either by passing a full dictionary, or by
# passing single arguments.

# spykingcircus spike sorting
default_SC_params['detect_threshold'] = 4

# parameters set by params dictionary
sorting_SC = ss.run_spykingcircus(recording=recording, **default_SC_params,
                                   output_folder='tmp_SC')
##############################################################################

# parameters set by params dictionary
sorting_SC_8 = ss.run_spykingcircus(recording=recording, detect_threshold=8.,
                                      output_folder='tmp_SC_8')

##############################################################################

print('Units found with threshold = 4:', sorting_SC.get_unit_ids())
print('Units found with threshold = 10:', sorting_SC_8.get_unit_ids())

##############################################################################
# Run other sorters
# ------------------
# 
# Some sorters (kilosort, ironclust, hdsort, ...) need to manually set the path to the source folder

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
