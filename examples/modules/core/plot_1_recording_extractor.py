'''
RecordingExtractor objects
==========================

The :code:`RecordingExtractor` is the basic class for handling recorded data.
Here is how it works.

A RecordingExtractor handle:

  * retrieve traces buffer across segments
  * dump/load from json/dict
  * cache (copy)


'''
import matplotlib.pyplot as plt

import numpy as np
import spikeinterface.extractors as se

##############################################################################
# We will create a :code:`RecordingExtractor` object from scratch using :code:`numpy` and the
# :code:`NumpyRecording`
#
# Let's define the properties of the dataset

num_channels = 7
sampling_frequency = 30000.  # in Hz
durations = [10., 15.] # in s for 2 segments
num_segments = 2
num_timepoints = [int(sampling_frequency * d) for d in durations]

##############################################################################
# We can generate a pure-noise timeseries dataset for 2 segments with 2 durations

traces0 = np.random.normal(0, 10, (num_timepoints[0], num_channels))
traces1 = np.random.normal(0, 10, (num_timepoints[1], num_channels))

##############################################################################
# And instantiate a :code:`NumpyRecording`:
# The object have a pretty print to summary it

recording = se.NumpyRecording(traces_list=[traces0, traces1], sampling_frequency=sampling_frequency)
print(recording)

##############################################################################
# We can now print properties that the :code:`RecordingExtractor` retrieves from the underlying recording.

print('Num. channels = {}'.format(len(recording.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording.get_sampling_frequency()))
print('Num. timepoints seg0= {}'.format(recording.get_num_segments()))
print('Num. timepoints seg0= {}'.format(recording.get_num_frames(segment_index=0)))
print('Num. timepoints seg1= {}'.format(recording.get_num_frames(segment_index=1)))


##############################################################################
# The geometry of the Probe is handle with the probeinterface 
#  lets generate a linear probe

from probeinterface import generate_linear_probe
from probeinterface.plotting import plot_probe
probe = generate_linear_probe(num_elec=7, ypitch=20, contact_shapes='circle', contact_shape_params={'radius': 6})

# probe have to be wiring to the recording
probe.set_device_channel_indices(np.arange(7))

recording = recording.set_probe(probe)
plot_probe(probe)


##############################################################################
# Some extractors also implement a :code:`write` function. 

# TODO make an example with NWB here instead of this
files_path=['traces0.raw', 'traces1.raw']
se.BinaryRecordingExtractor.write_recording(recording, files_path)



##############################################################################
# and read it back with the proper extractor:
# not that this new recording is now "on disk" and not "in memory"
# also note that it is a lazy loading the file is not read

recording2 = se.BinaryRecordingExtractor(files_path,  sampling_frequency, num_channels, traces0.dtype)
print(recording2)

##############################################################################
# reading traces buffer is done on demand

# entire segment 0
taces0 = recording2.get_traces(segment_index=0)
# ârt of segment 1
taces1_short = recording2.get_traces(segment_index=1, end_frame=50)
print(traces0.shape)
print(taces1_short.shape)


##############################################################################
# A recording have internaly a channel_ids
# this a vector that can have dtype int or str

print('chan_ids (dtype=int):', recording.get_channel_ids())

recording3 = se.NumpyRecording(traces_list=[traces0, traces1],
                                                            sampling_frequency=sampling_frequency,
                                                            channel_ids=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
print('chan_ids (dtype=str):', recording3.get_channel_ids())


##############################################################################
#  theses channel_ids are used when you want get only some channels


traces = recording3.get_traces(segment_index=1, end_frame=50, channel_ids=['a', 'd'])
print(traces.shape)

##############################################################################
# You can also get a channel slice view of the recording

recording4 = recording3.channel_slice(channel_ids=['a', 'c', 'e'])
print(recording4)
print(recording4.get_channel_ids())

# which is equivalent to 
from spikeinterface import ChannelSliceRecording
recording4 = ChannelSliceRecording(recording3, channel_ids=['a', 'c', 'e'])

##############################################################################
# Here an example on how to split channels on a particular property

recording3.set_property('group', [0, 0, 0, 1, 1, 1, 2])

recordings = recording3.split_by(property='group')
print(recordings)
print(recordings[0].get_channel_ids())
print(recordings[1].get_channel_ids())
print(recordings[2].get_channel_ids())


###############################################################################
# A recording can be "dump" (export)
#  * a dict
#  * a json file
#  * a pickle file
# 
#  A dump is lazy, the traces are not exported. Only kwargs and property that make possible the recording to
# be reconstructed

from spikeinterface import load_extractor
from pprint import pprint

d = recording2.to_dict()
pprint(d)

recording2_loaded = load_extractor(d)
print(recording2_loaded)

###############################################################################
#  Same for JSON. Persistent on disk

recording2.dump('my_recording.json')

recording2_loaded = load_extractor('my_recording.json')
print(recording2_loaded)

###############################################################################
# note that dump to not copy the buffer to disk
# If you want to also make traces persistent you need to used save()
#  this of course use more ressource.
# This operation is very usefull to saved long computation.

recording2.save(folder='./my_recording')

import os
pprint(os.listdir('./my_recording'))

recording2_cached = load_extractor('my_recording.json')
print(recording2_cached)










