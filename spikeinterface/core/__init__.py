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

# generator of simple object for testing or examples
from .generate import (generate_recording, generate_sorting,
  create_sorting_npz, generate_snippets,
  synthesize_random_firings,  inject_some_duplicate_units,
  inject_some_split_units, synthetize_spike_train_bad_isi)

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
    select_segment_sorting,
    SelectSegmentSorting
)

# default folder
from .globals import (set_global_tmp_folder, get_global_tmp_folder,
                      is_set_global_tmp_folder, reset_global_tmp_folder,
                      get_global_dataset_folder, set_global_dataset_folder,
                      is_set_global_dataset_folder,
                      get_global_job_kwargs, set_global_job_kwargs, reset_global_job_kwargs)

# tools 
from .core_tools import write_binary_recording, write_to_h5_dataset_format, write_binary_recording, read_python, \
    write_python
from .job_tools import ensure_n_jobs, ensure_chunk_size, ChunkRecordingExecutor, split_job_kwargs, fix_job_kwargs
from .recording_tools import (get_random_data_chunks, get_channel_distances, get_closest_channels,
                              get_noise_levels, get_chunk_with_margin, order_channels_by_depth)
from .waveform_tools import extract_waveforms_to_buffers
from .snippets_tools import snippets_from_sorting

# waveform extractor
from .waveform_extractor import WaveformExtractor, extract_waveforms, load_waveforms, precompute_sparsity

# retrieve datasets
from .datasets import download_dataset

from .old_api_utils import (create_recording_from_old_extractor, create_sorting_from_old_extractor,
                            create_extractor_from_new_recording, create_extractor_from_new_sorting)

# templates addition
from .injecttemplates import InjectTemplatesRecording, InjectTemplatesRecordingSegment, inject_templates

# template tools
from .template_tools import (
    get_template_amplitudes,
    get_template_extremum_channel,
    get_template_extremum_channel_peak_shift,
    get_template_extremum_amplitude,
    get_template_channel_sparsity,
)

# channel sparsity
from .sparsity import ChannelSparsity, compute_sparsity