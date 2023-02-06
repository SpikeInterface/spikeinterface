"""
Run spike sorting on concatenated recordings
============================================

In several experiments, several recordings are performed in sequence, for example a baseline/intervention.
In these cases, since the underlying spiking activity can be assumed to be the same (or at least very similar), the
recordings can be concatenated. This notebook shows how to concatenate the recordings before spike sorting and how to
split the sorted output based on the concatenation.

Note that some sorters (tridesclous, ...) handle directly multi segment paradigm, in that case we will use the
:py:func:`~spikeinterface.core.append_recordings()` function. Many sorters do not handle multi segment, in that case
we will use the :py:func:`~spikeinterface.core.concatenate_recordings()` function.


See also "Append and/or concatenate segments" in core tutorials.

"""

import spikeinterface.full as si

##############################################################################
#  When performing an experiment with multiple consecutive recordings, it can be a good idea to concatenate the single
# recordings, as this can improve the spike sorting performance and it doesn't require to track the neurons over the
# different recordings.
#
# Let's create a toy example with 4 channels (the :code:`dumpable=True` dumps the extractors to a file, which is
# required for parallel sorting):

recording_single, _ = si.toy_example(duration=10, num_channels=4, seed=0, num_segments=1)
print(recording_single)

# make dumpable
recording_single = recording_single.save()

##############################################################################
# Let's now assume that we have 4 recordings. In our case we will concatenate the :code:`recording_single` 4 times. We
# first need to build a list of :code:`RecordingExtractor` objects:

recordings_list = []
for i in range(4):
    recordings_list.append(recording_single)


##############################################################################
# Case 1. : the sorter handle multi segment

multirecording = si.append_recordings(recordings_list)
# lets put a probe
multirecording = multirecording.set_probe(recording_single.get_probe())
print(multirecording)

# run tridesclous in multi segment mode
multisorting = si.run_tridesclous(multirecording)
print(multisorting)

##############################################################################
# Case 2. : the sorter DO NOT handle multi segment
# In that case the `concatenate_recordings()` mimic a mono segment that concatenate all segment

multirecording = si.concatenate_recordings(recordings_list)
# lets put a probe
multirecording = multirecording.set_probe(recording_single.get_probe())
print(multirecording)

# run klusta in mono segment mode
# multisorting = si.run_klusta(multirecording)


