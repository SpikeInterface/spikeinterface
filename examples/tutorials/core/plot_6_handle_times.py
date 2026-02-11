"""
Handle time information
=======================

By default, SpikeInterface assumes that a recording is uniformly sampled and it starts at 0 seconds.
However, in some cases there could be a different start time or even some missing frames in the recording.

This notebook shows how to handle time information in SpikeInterface recording and sorting objects.
"""

from spikeinterface.extractors import toy_example

##############################################################################
# First let's generate a toy example with a single segment:

rec, sort = toy_example(num_segments=1)


##############################################################################
# Generally, the time information would be automatically loaded when reading a
# recording.
# However, sometimes we might need to add a time vector externally.
# For example, let's create a time vector by getting the default times and
# adding 5 s:

default_times = rec.get_times()
print(default_times[:10])
new_times = default_times + 5

##############################################################################
# We can now set the new time vector with the :code:`set_times()` function.
# Additionally, we can register to recording object to the sorting one so that
# time information can be accessed by the sorting object as well (note that this
# link is lost in case the sorting object is saved to disk!):

rec.set_times(new_times)
sort.register_recording(rec)

# print new times
print(rec.get_times()[:10])

# print spike times (internally uses registered recording times)
spike_times0 = sort.get_unit_spike_train(sort.unit_ids[0], return_times=True)
print(spike_times0[:10])


##############################################################################
# While here we have shown how to set times only for a mono-segment recording,
# times can also be handled in multi-segment recordings (using the
# :code:`segment_index` argument when calling :code:`set_times()`).
#
# Finally, you you run spike sorting through :code:`spikeinterface`, the recording
# is automatically registered to the output sorting object!
