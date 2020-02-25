"""
Run spike sorting by property
=============================

Sometimes you may want to spike sort different electrodes separately. For example your probe can have several channel
groups (for example tetrodes) or you might want to spike sort different brain regions separately, In these cases, you
can spike sort by property.
"""

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import time

##############################################################################
#  Sometimes, you might want to sort your data depending on a specific property of your recording channels.
#  
# For example, when using multiple tetrodes, a good idea is to sort each tetrode separately. In this case, channels
# belonging to the same tetrode will be in the same 'group'. Alternatively, for long silicon probes, such as
# Neuropixels, you could sort different areas separately, for example hippocampus and thalamus.
#  
# All this can be done by sorting by 'property'. Properties can be loaded to the recording channels either manually
# (using the :code:`set_channel_property` method), or by using a probe file. In this example we will create a 16 channel
# recording and split it in four channel groups (tetrodes).
#
# Let's create a toy example with 16 channels:

recording_tetrodes, sorting_true = se.example_datasets.toy_example(duration=10, num_channels=16)

##############################################################################
# Initially there is no group information ('location' is loaded automatically when creating toy data):

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

##############################################################################
# We can load 'group' information using the '.prb' file:

recording_tetrodes = recording_tetrodes.load_probe_file('tetrode_16.prb')
print(recording_tetrodes.get_shared_channel_property_names())

##############################################################################
# We can now use the launcher to spike sort by the property 'group'. The different groups can also be sorted in
# parallel, and the output sorting extractor will have the same property used for sorting. Running in parallel
# (in separate threads) can speed up the computations.
#
# Let's first run the four channel groups sequentially:

t_start = time.time()
sorting_tetrodes = ss.run_sorter('klusta', recording_tetrodes, output_folder='tmp_tetrodes',
                                 grouping_property='group', parallel=False, verbose=False)
print('Elapsed time: ', time.time() - t_start)

##############################################################################
# then in parallel:

t_start = time.time()
sorting_tetrodes_p = ss.run_sorter('klusta', recording_tetrodes, output_folder='tmp_tetrodes_par',
                                   grouping_property='group', parallel=True, verbose=False)
print('Elapsed time parallel: ', time.time() - t_start)

##############################################################################
# The units of the sorted output will have the same property used for spike sorting:

print(sorting_tetrodes.get_shared_unit_property_names())

##############################################################################
# Note that channels can be split by any property. Let's for example assume that half of the tetrodes are in hippocampus
# CA1 region, and the other half is in CA3. first we have to load this property (this can be done also from the '.prb'
# file):

for ch in recording_tetrodes.get_channel_ids()[:int(recording_tetrodes.get_num_channels() / 2)]:
    recording_tetrodes.set_channel_property(ch, property_name='region', value='CA1')

for ch in recording_tetrodes.get_channel_ids()[int(recording_tetrodes.get_num_channels() / 2):]:
    recording_tetrodes.set_channel_property(ch, property_name='region', value='CA3')

for ch in recording_tetrodes.get_channel_ids():
    print(recording_tetrodes.get_channel_property(ch, property_name='region'))

##############################################################################
# Now let's spike sort by 'region' and check that the units of the sorted output have this property:

sorting_regions = ss.run_sorter('klusta', recording_tetrodes, output_folder='tmp_regions',
                                grouping_property='region', parallel=True)

print(sorting_regions.get_shared_unit_property_names())
