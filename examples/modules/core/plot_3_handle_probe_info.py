'''
Handling probe information
===========================

In order to properly spike sort, you may need to load information related to the probe you are using.

spikeinterface use internaly `probe interface <https://probeinterface.readthedocs.io/>`_ to handle
probe or probe group for recordings.

Depending the dataset the `Probe` object can be already include or you have to settlt it manually.

Here's how!
'''
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as se

##############################################################################
# First, let's create a toy example:

recording, sorting_true = se.toy_example(duration=10, num_channels=32, seed=0, num_segments=2)
print(recording)

###############################################################################
# This genertor already contain a probe object you can retreive directly an plot

probe = recording.get_probe()
print(probe)
from probeinterface.plotting import plot_probe
plot_probe(probe)


###############################################################################
#  You can also change the probe 
# In that case you need to manually make the wiring
# Lets use a probe from cambridgeneurotech with 32ch

from probeinterface import get_probe
other_probe = get_probe('cambridgeneurotech', 'ASSY-37-E-1')
print(other_probe)
other_probe.set_device_channel_indices(np.arange(32))
recording_2_shanks = recording.set_probe(other_probe, group_mode='by_shank')
plot_probe(recording_2_shanks.get_probe())


###############################################################################
# Now let's check what we have loaded.
# The `group_mode='by_shank'`  have set the 'group' property automatically
# we can use it to split the recording in 2 small ones

print(recording_2_shanks)
print(recording_2_shanks.get_property('group'))

rec0, rec1 = recording_2_shanks.split_by(property='group')
print(rec0)
print(rec1)

###############################################################################
# Note that some format (mearec, spikeglx) handle directly the probe geometry
# for almost all other format the probe and the wiring have to be set manually with the probeinterface library
