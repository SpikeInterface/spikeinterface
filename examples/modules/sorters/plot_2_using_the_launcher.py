"""
Use the spike sorting launcher
==============================

This example shows how to use the spike sorting launcher. The launcher allows to parameterize the sorter name and
to run several sorters on one or multiple recordings.

"""

import spikeinterface.extractors as se
import spikeinterface.sorters as ss

##############################################################################
# First, let's create the usual toy example:

recording, sorting_true = se.toy_example(duration=10, seed=0, num_segments=1)
print(recording)
print(sorting_true)

##############################################################################
# Lets cache this recording to make it "dumpable"

recording = recording.save(name='toy')
print(recording)

##############################################################################
# The launcher enables to call any spike sorter with the same functions:  :code:`run_sorter` and :code:`run_sorters`.
# For running multiple sorters on the same recording extractor or a collection of them, the :code:`run_sorters`
# function can be used.
#
# Let's first see how to run a single sorter, for example, Klusta:

# The sorter name can be now a parameter, e.g. chosen with a command line interface or a GUI
sorter_name = 'herdingspikes'
sorting_HS = ss.run_sorter(sorter_name='herdingspikes', recording=recording, output_folder='my_sorter_output', clustering_bandwidth=8)
print(sorting_HS.get_unit_ids())

##############################################################################
#
# You can also run multiple sorters on the same recording:

recordings = {'toy' : recording }
sorter_list = ['herdingspikes', 'tridesclous']
sorter_params = { 'herdingspikes': {'clustering_bandwidth' : 8} }
sorting_output = ss.run_sorters(sorter_list, recordings, working_folder='tmp_some_sorters', 
                                mode_if_folder_exists='overwrite', sorter_params=sorter_params)

##############################################################################
# The 'mode' argument allows to 'overwrite' the 'working_folder' (if existing), 'raise' and Exception, or 'keep' the
# folder and skip the spike sorting run.
#
# To 'sorting_output' is a dictionary that has (recording, sorter) pairs as keys and the correspondent
# :code:`SortingExtractor` as values. It can be accessed as follows:

for (rec_name, sorter_name), sorting in sorting_output.items():
    print(rec_name, sorter_name, ':', sorting.get_unit_ids())

##############################################################################
# With the same mechanism, you can run several spike sorters on many recordings, just by creating a list/dict of
# :code:`RecordingExtractor` objects (:code:`recording_list`).
