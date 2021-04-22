"""
Run spike sorting on concatenated recordings
============================================

In several experiments, several recordings are performed in sequence, for example a baseline/intervention.
In these cases, since the underlying spiking activity can be assumed to be the same (or at least very similar), the
recordings can be concatenated. This notebook shows how to concatenate the recordings before spike sorting and how to
split the sorted output based on the concatenation.
"""

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import time

##############################################################################
#  When performing an experiment with multiple consecutive recordings, it can be a good idea to concatenate the single
# recordings, as this can improve the spike sorting performance and it doesn't require to track the neurons over the
# different recordings.
#  
# This can be done very easily in SpikeInterface using a combination of the :code:`MultiRecordingTimeExtractor` and the
# :code:`SubSortingExtractor` objects.
#
# Let's create a toy example with 4 channels (the :code:`dumpable=True` dumps the extractors to a file, which is
# required for parallel sorting):

recording_single, _ = se.example_datasets.toy_example(duration=10, num_channels=4, dumpable=True)

##############################################################################
# Let's now assume that we have 4 recordings. In our case we will concatenate the :code:`recording_single` 4 times. We
# first need to build a list of :code:`RecordingExtractor` objects:

recordings_list = []
for i in range(4):
    recordings_list.append(recording_single)


##############################################################################
# We can now use the :code:`recordings_list` to instantiate a :code:`MultiRecordingTimeExtractor`, which concatenates
# the traces in time:

multirecording = se.MultiRecordingTimeExtractor(recordings=recordings_list)

##############################################################################
# Since the :code:`MultiRecordingTimeExtractor` is a :code:`RecordingExtractor`, we can run spike sorting "normally"

multisorting = ss.run_klusta(multirecording)

##############################################################################
# The returned :code:`multisorting` object is a normal :code:`SortingExtractor`, but we now that its spike trains are
# concatenated similarly to the recording concatenation. So we have to split them back. We can do that using the `epoch`
# information in the :code:`MultiRecordingTimeExtractor`:

sortings = []

sortings = []
for epoch in multisorting.get_epoch_names():
    info = multisorting.get_epoch_info(epoch)
    sorting_single = se.SubSortingExtractor(multisorting, start_frame=info['start_frame'], end_frame=info['end_frame'])
    sortings.append(sorting_single)

##############################################################################
# The :code:`SortingExtractor` objects in  the :code:`sortings` list contain now split spike trains. The nice thing of
# this approach is that the unit_ids for the different epochs are the same unit!
