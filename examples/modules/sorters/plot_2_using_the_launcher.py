"""
Use the spike sorting launcher
==============================

This example shows how to use the spike sorting launcher. The launcher allows to parameterize the sorter name and
to run several sorters on one or multiple recordings.

"""

import spikeinterface.extractors as se
import spikeinterface.sorters as ss

##############################################################################
# First, let's create the usueal toy example:

recording, sorting_true = se.example_datasets.toy_example(duration=60, seed=0)

##############################################################################
# The launcher enables to call any spike sorter with the same functions:  :code:`run_sorter` and :code`run_sorters`.
# For running multiple sorters on the same recording extractor or a collection of them, the :code:`run_sorters`
# function can be used.
#
# Let's first see how to run a single sorter, for example, Klusta:

# The sorter name can be now a parameter, e.g. chosen with a command line interface or a GUI
sorter_name = 'klusta'
sorting_KL = ss.run_sorter(sorter_name_or_class='klusta', recording=recording, output_folder='my_sorter_output')
print(sorting_KL.get_unit_ids())

##############################################################################
# This will launch the klusta sorter on the recording object.
#
# You can also run multiple sorters on the same recording:

recording_list = [recording]
sorter_list = ['klusta', 'mountainsort4', 'tridesclous']
sorting_output = ss.run_sorters(sorter_list, recording_list, working_folder='tmp_some_sorters', mode='overwrite')

##############################################################################
# The 'mode' argument allows to 'overwrite' the 'working_folder' (if existing), 'raise' and Exception, or 'keep' the
# folder and skip the spike sorting run.
#
# To 'sorting_output' is a dictionary that has (recording, sorter) pairs as keys and the correspondent
# :code:`SortingExtractor` as values. It can be accessed as follows:

for (rec, sorter), sort in sorting_output.items():
    print(rec, sorter, ':', sort.get_unit_ids())

##############################################################################
# With the same mechanism, you can run several spike sorters on many recordings, just by creating a list of
# :code:`RecordingExtractor` objects (:code:`recording_list`).
