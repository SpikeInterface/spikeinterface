"""
Contain core class:
  * Recording
  * Sorting

And contain also "core extractors" used for caching:
  * BinaryRecordingExtractor
  * NpzSortingExtractor

"""
from .recording import Recording, RecordingSegment
from .sorting import Sorting, SortingSegment

# from .types import Order, ChannelIndex, SampleIndex, SamplingFrequencyHz