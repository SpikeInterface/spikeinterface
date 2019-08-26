"""
Run sorters
=======

This is an example.

"""


##############################################################################
# Check available sorters
# ------------------------------
# 

import spikeinterface.sorters as sorters

print(sorters.available_sorters())


##############################################################################
# This will list the sorters installed in the machine. Each spike sorter
# is implemented in a class. To access the class names you can run:

print(sorters.installed_sorter_list)


##############################################################################
# Check and set sorters parameters
# ---------------------------------------------------
# 

default_ms4_params = sorters.Mountainsort4Sorter.default_params()
print(default_ms4_params)