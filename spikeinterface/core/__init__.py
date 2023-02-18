from .base import (
    load_extractor,  # , load_extractor_from_dict, load_extractor_from_json, load_extractor_from_pickle
)
from .baseevent import BaseEvent, BaseEventSegment
from .baserecording import BaseRecording, BaseRecordingSegment
from .baserecordingsnippets import BaseRecordingSnippets
from .basesnippets import BaseSnippets, BaseSnippetsSegment
from .basesorting import BaseSorting, BaseSortingSegment
from .binaryfolder import BinaryFolderRecording, read_binary_folder

# main extractor from dump and cache
from .binaryrecordingextractor import BinaryRecordingExtractor, read_binary
from .channelsaggregationrecording import (
    ChannelsAggregationRecording,
    aggregate_channels,
)

# utility extractors (equivalent to OLD subrecording/subsorting)
from .channelslice import ChannelSliceRecording, ChannelSliceSnippets

# tools
from .core_tools import (
    read_python,
    write_binary_recording,
    write_python,
    write_to_h5_dataset_format,
)

# retrieve datasets
from .datasets import download_dataset
from .frameslicerecording import FrameSliceRecording
from .frameslicesorting import FrameSliceSorting

# generator of simple object for testing or examples
from .generate import (
    create_sorting_npz,
    generate_recording,
    generate_snippets,
    generate_sorting,
    inject_some_duplicate_units,
    inject_some_split_units,
    synthesize_random_firings,
    synthetize_spike_train_bad_isi,
)

# default folder
from .globals import (
    get_global_dataset_folder,
    get_global_job_kwargs,
    get_global_tmp_folder,
    is_set_global_dataset_folder,
    is_set_global_tmp_folder,
    reset_global_job_kwargs,
    reset_global_tmp_folder,
    set_global_dataset_folder,
    set_global_job_kwargs,
    set_global_tmp_folder,
)

# templates addition
from .injecttemplates import (
    InjectTemplatesRecording,
    InjectTemplatesRecordingSegment,
    inject_templates,
)
from .job_tools import (
    ChunkRecordingExecutor,
    ensure_chunk_size,
    ensure_n_jobs,
    fix_job_kwargs,
    split_job_kwargs,
)
from .npyfoldersnippets import NpyFolderSnippets, read_npy_snippets_folder
from .npysnippetsextractor import NpySnippetsExtractor, read_npy_snippets
from .npzfolder import NpzFolderSorting, read_npz_folder
from .npzsortingextractor import NpzSortingExtractor, read_npz_sorting
from .numpyextractors import NumpyEvent, NumpyRecording, NumpySnippets, NumpySorting
from .old_api_utils import (
    create_extractor_from_new_recording,
    create_extractor_from_new_sorting,
    create_recording_from_old_extractor,
    create_sorting_from_old_extractor,
)
from .recording_tools import (
    get_channel_distances,
    get_chunk_with_margin,
    get_closest_channels,
    get_noise_levels,
    get_random_data_chunks,
    order_channels_by_depth,
)

# utils to append and concatenate segment (equivalent to OLD MultiRecordingTimeExtractor)
from .segmentutils import (
    AppendSegmentRecording,
    AppendSegmentSorting,
    ConcatenateSegmentRecording,
    SelectSegmentRecording,
    SelectSegmentSorting,
    SplitSegmentSorting,
    append_recordings,
    append_sortings,
    concatenate_recordings,
    select_segment_recording,
    select_segment_sorting,
    split_recording,
    split_sorting,
)
from .snippets_tools import snippets_from_sorting

# channel sparsity
from .sparsity import ChannelSparsity, compute_sparsity

# template tools
from .template_tools import (
    get_template_amplitudes,
    get_template_channel_sparsity,
    get_template_extremum_amplitude,
    get_template_extremum_channel,
    get_template_extremum_channel_peak_shift,
)
from .unitsaggregationsorting import UnitsAggregationSorting, aggregate_units
from .unitsselectionsorting import UnitsSelectionSorting

# waveform extractor
from .waveform_extractor import (
    BaseWaveformExtractorExtension,
    WaveformExtractor,
    extract_waveforms,
    load_waveforms,
    precompute_sparsity,
)
from .waveform_tools import extract_waveforms_to_buffers
from .zarrrecordingextractor import (
    ZarrRecordingExtractor,
    get_default_zarr_compressor,
    read_zarr,
)
