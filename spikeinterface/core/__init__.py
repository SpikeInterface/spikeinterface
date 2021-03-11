"""
Contain core class:
  * Recording
  * Sorting

And contain also "core extractors" used for caching:
  * BinaryRecordingExtractor
  * NpzSortingExtractor

"""
from .base import load_extractor #, load_extractor_from_dict, load_extractor_from_json, load_extractor_from_pickle
from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, BaseSortingSegment

# main extractor from dump and cache
from .binaryrecordingextractor import BinaryRecordingExtractor
from .npzsortingextractor import NpzSortingExtractor

# utility extractors
from .channelslicerecording import ChannelSliceRecording

# default folder
from .default import (set_global_tmp_folder,
        get_global_tmp_folder, is_set_global_tmp_folder)
