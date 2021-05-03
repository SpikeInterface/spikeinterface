"""
Append and/or concatenate segments
===========================

Sometimes a recording can be splitted in several sequence, for instance because aseline/intervention/pauses.

The terminology in spikeniterface is "segment" (like in neo).

spikeinterface have tools to manipulate theses segments. There are two ways:

  1. append_recordings()/append_sortings()/append_events()
  2. concatenate_recordings()/ concatenate_sortings()/ concatenate_events()

Imagine we have 2 recordings with respectively 2 and 3 segments:

  1. In case 1 we will have one recording with 5 segments
  2. In case 2 we will have one recording with 1 "big" segment that virtually concatenate all segments.

Here a short example.
"""

import matplotlib.pyplot as plt
import numpy as np


from spikeinterface import NumpyRecording, NumpySorting
from spikeinterface import append_recordings, concatenate_recordings

##############################################################################
# First let's generate 2 recordings with 2 and 3 segments respectively.
# 

sampling_frequency = 1000.

trace0 = np.zeros((150, 5), dtype='float32')
trace1 = np.zeros((100, 5), dtype='float32')
rec0 = NumpyRecording([trace0, trace1], sampling_frequency)
print(rec0)

trace2 = np.zeros((50, 5), dtype='float32')
trace3 = np.zeros((200, 5), dtype='float32')
trace4 = np.zeros((120, 5), dtype='float32')
rec1 = NumpyRecording([trace2, trace3, trace4], sampling_frequency)
print(rec1)


##############################################################################
# lets use `append_recordings()`

recording_list = [rec0, rec1]
rec = append_recordings(recording_list)
print(rec)
for i in range(rec.get_num_segments()):
    s = rec.get_num_samples(segment_index=i)
    print(f'segment {i} num_samples {s}')

##############################################################################
# lets use `concatenate_recordings()`

recording_list = [rec0, rec1]
rec = concatenate_recordings(recording_list)
print(rec)
s = rec.get_num_samples(segment_index=0)
print(f'segment {0} num_samples {s}')


