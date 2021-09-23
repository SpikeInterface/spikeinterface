"""
Run spike sorting by property
=============================

Sometimes you may want to spike sort different electrodes separately. For example your probe can have several channel
groups (for example tetrodes) or you might want to spike sort different brain regions separately, In these cases, you
can spike sort by property.
"""

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.sorters as ss


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
# Let's create a toy example with 16 channels (the :code:`dumpable=True` dumps the extractors to a file, which is
# required for parallel sorting):

recording, sorting_true = se.toy_example(duration=[10.], num_segments=1, num_channels=16)
# make dumpable
recording= recording.save()

##############################################################################
# Initially all channel are in the same group.

print(recording.get_channel_groups())

##############################################################################
# Lets now change the probe mapping and assign a 4 tetrodes to this recording.
# for this we will use the `probeinterface` module and create a `ProbeGroup` containing for dummy tetrode.

from probeinterface import generate_tetrode, ProbeGroup

probegroup = ProbeGroup()
for i in range(4):
    tetrode = generate_tetrode()
    tetrode.set_device_channel_indices(np.arange(4) + i * 4)
    probegroup.add_probe(tetrode)

##############################################################################
#  now our new recording contain 4 groups

recording_4_tetrodes = recording.set_probegroup(probegroup, group_mode='by_probe')

# get group
print(recording_4_tetrodes.get_channel_groups())
# similar to this
print(recording_4_tetrodes.get_property('group'))

##############################################################################
#  this "group" property can be use to split our new recording into 4 recording
# we get a list of 4 ChannelSliceRecording (ex "sub-recording")
# This is done without any copy, each ChannelSliceRecording is a view of the parent recording
#  Note that here we use 'group' for splitting but it could be done on any property.

recordings = recording_4_tetrodes.split_by(property='group')
print(recordings)

##############################################################################
# We can also get a dict instead of the list which is easier to handle group keys.

recordings = recording_4_tetrodes.split_by(property='group', outputs='dict')
print(recordings)


##############################################################################
# We can now use the `run_sorters()` function instead of the `run_sorter()`.
# This function can run several sorters on several recording with different parallel engines.
#  here we use engine 'loop' but we could use also  'joblib' or 'dask' for multi process or multi node computing.
#  have a look to the documentation of this function that handle many cases.

sorter_list = ['tridesclous']
working_folder = 'sorter_outputs'
results = ss.run_sorters(sorter_list, recordings, working_folder,
            engine='loop', with_output=True, mode_if_folder_exists='overwrite')

##############################################################################
#  the output is a dict with all combinations of (group, sorter_name)

from pprint import pprint
pprint(results)
