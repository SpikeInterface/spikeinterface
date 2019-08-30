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

import spikeextractors as se
import spikeinterface.sorters as ss

##############################################################################
# First, let's create a toy example:

recording, sorting_true = se.example_datasets.toy_example(duration=60, seed=0)

##############################################################################
# 4) Use the spike sorter launcher
# --------------------------------
# The launcher enables to call any spike sorter with the same functions:
# ``run_sorter`` and ``run_sorters``. For running multiple sorters on the
# same recording extractor or a collection of them, the ``run_sorters``
# function can be used.

recording_list = [recording]
sorter_list = ['klusta', 'mountainsort4', 'tridesclous']
sorting_output = ss.run_sorters(sorter_list, recording_list, working_folder='tmp_some_sorters')

##############################################################################
# retrieve results for all (recording, sorter) pairs
# access with dict with tuple of keys


for (rec, sorter), extractor in sorting_output.items():
    print(rec, sorter, ':', extractor.get_unit_ids())
