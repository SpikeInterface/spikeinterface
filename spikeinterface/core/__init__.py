"""
Contain core class:
  * Recording
  * Sorting

And contain also "core extractors" used for caching:
  * BinaryRecordingExtractor
  * NpzSortingExtractor

"""
from .base import load_extractor  # , load_extractor_from_dict, load_extractor_from_json, load_extractor_from_pickle
from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, BaseSortingSegment
from .baseevent import BaseEvent, BaseEventSegment
from .basesnippets import BaseSnippets, BaseSnippetsSegment
from .baserecordingsnippets import BaseRecordingSnippets

# main extractor from dump and cache
from .binaryrecordingextractor import BinaryRecordingExtractor, read_binary
from .npzsortingextractor import NpzSortingExtractor, read_npz_sorting
from .numpyextractors import NumpyRecording, NumpySorting, NumpyEvent, NumpySnippets
from .zarrrecordingextractor import ZarrRecordingExtractor, read_zarr, get_default_zarr_compressor
from .binaryfolder import BinaryFolderRecording, read_binary_folder
from .npysnippetsextractor import NpySnippetsExtractor
from .npyfoldersnippets import NpyFolderSnippets, read_npy_snippets_folder

# utility extractors (equivalent to OLD subrecording/subsorting)
from .channelslice import ChannelSliceRecording, ChannelSliceSnippets
from .unitsselectionsorting import UnitsSelectionSorting
from .frameslicerecording import FrameSliceRecording
from .frameslicesorting import FrameSliceSorting

from .channelsaggregationrecording import ChannelsAggregationRecording, aggregate_channels
from .unitsaggregationsorting import UnitsAggregationSorting, aggregate_units

# utils to append and concatenate segment (equivalent to OLD MultiRecordingTimeExtractor)
from .segmentutils import (
    append_recordings,
    AppendSegmentRecording,
    concatenate_recordings,
    ConcatenateSegmentRecording,
    split_recording,
    select_segment_recording,
    SelectSegmentRecording,
    append_sortings,
    AppendSegmentSorting,
    split_sorting,
    SplitSegmentSorting,
)

# default folder
from .default_folders import (set_global_tmp_folder, get_global_tmp_folder,
                              is_set_global_tmp_folder, reset_global_tmp_folder,
                              get_global_dataset_folder, set_global_dataset_folder, is_set_global_dataset_folder)

# tools 
from .core_tools import write_binary_recording, write_to_h5_dataset_format, write_binary_recording, read_python, \
    write_python
from .job_tools import ensure_n_jobs, ensure_chunk_size, ChunkRecordingExecutor
from .recording_tools import (get_random_data_chunks, get_channel_distances, get_closest_channels, 
                              get_noise_levels, get_chunk_with_margin)
from .waveform_tools import extract_waveforms_to_buffers
from .snippets_tools import snippets_from_sorting

# waveform extractor
from .waveform_extractor import WaveformExtractor, extract_waveforms

# retrieve datasets
from .datasets import download_dataset

from .old_api_utils import (create_recording_from_old_extractor, create_sorting_from_old_extractor,
                            create_extractor_from_new_recording, create_extractor_from_new_sorting)