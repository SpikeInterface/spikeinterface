from __future__ import annotations

from typing import Type

# most important extractor are in spikeinterface.core
from spikeinterface.core import (
    BaseRecording,
    BaseSorting,
    BinaryFolderRecording,
    BinaryRecordingExtractor,
    NumpyRecording,
    NpzSortingExtractor,
    NumpySorting,
    NpySnippetsExtractor,
    ZarrRecordingExtractor,
    ZarrSortingExtractor,
    read_binary,
    read_zarr,
    read_npz_sorting,
)

# sorting/recording/event from neo
from .neoextractors import *

# non-NEO objects implemented in neo folder
from .neoextractors import NeuroScopeSortingExtractor, MaxwellEventExtractor

# NWB sorting/recording/event
from .nwbextractors import (
    NwbRecordingExtractor,
    NwbSortingExtractor,
    NwbTimeSeriesExtractor,
    read_nwb,
    read_nwb_recording,
    read_nwb_sorting,
    read_nwb_timeseries,
)

from .cbin_ibl import CompressedBinaryIblExtractor, read_cbin_ibl
from .iblextractors import IblRecordingExtractor, IblSortingExtractor, read_ibl_recording, read_ibl_sorting
from .mcsh5extractors import MCSH5RecordingExtractor, read_mcsh5
from .whitematterrecordingextractor import WhiteMatterRecordingExtractor

# sorting extractors in relation with a sorter
from .cellexplorersortingextractor import CellExplorerSortingExtractor, read_cellexplorer
from .klustaextractors import KlustaSortingExtractor, read_klusta
from .hdsortextractors import HDSortSortingExtractor, read_hdsort
from .mclustextractors import MClustSortingExtractor, read_mclust
from .waveclustextractors import WaveClusSortingExtractor, read_waveclus
from .yassextractors import YassSortingExtractor, read_yass
from .combinatoextractors import CombinatoSortingExtractor, read_combinato
from .tridesclousextractors import TridesclousSortingExtractor, read_tridesclous
from .spykingcircusextractors import SpykingCircusSortingExtractor, read_spykingcircus
from .herdingspikesextractors import HerdingspikesSortingExtractor, read_herdingspikes
from .mdaextractors import MdaRecordingExtractor, MdaSortingExtractor, read_mda_recording, read_mda_sorting
from .phykilosortextractors import PhySortingExtractor, KiloSortSortingExtractor, read_phy, read_kilosort
from .sinapsrecordingextractors import (
    SinapsResearchPlatformRecordingExtractor,
    SinapsResearchPlatformH5RecordingExtractor,
    read_sinaps_research_platform,
    read_sinaps_research_platform_h5,
)

# sorting in relation with simulator
from .shybridextractors import (
    SHYBRIDRecordingExtractor,
    SHYBRIDSortingExtractor,
    read_shybrid_recording,
    read_shybrid_sorting,
)

# snippers
from .waveclussnippetstextractors import WaveClusSnippetsExtractor, read_waveclus_snippets


# misc
from .alfsortingextractor import ALFSortingExtractor, read_alf_sorting


########################################

recording_extractor_full_list = [
    BinaryFolderRecording,
    BinaryRecordingExtractor,
    ZarrRecordingExtractor,
    # natively implemented in spikeinterface.extractors
    NumpyRecording,
    SHYBRIDRecordingExtractor,
    MdaRecordingExtractor,
    NwbRecordingExtractor,
    # others
    CompressedBinaryIblExtractor,
    IblRecordingExtractor,
    MCSH5RecordingExtractor,
    SinapsResearchPlatformRecordingExtractor,
    WhiteMatterRecordingExtractor,
]
recording_extractor_full_list += neo_recording_extractors_list

sorting_extractor_full_list = [
    NpzSortingExtractor,
    ZarrSortingExtractor,
    NumpySorting,
    # natively implemented in spikeinterface.extractors
    MdaSortingExtractor,
    SHYBRIDSortingExtractor,
    ALFSortingExtractor,
    KlustaSortingExtractor,
    HDSortSortingExtractor,
    MClustSortingExtractor,
    WaveClusSortingExtractor,
    YassSortingExtractor,
    CombinatoSortingExtractor,
    TridesclousSortingExtractor,
    SpykingCircusSortingExtractor,
    HerdingspikesSortingExtractor,
    KiloSortSortingExtractor,
    PhySortingExtractor,
    NwbSortingExtractor,
    NeuroScopeSortingExtractor,
    IblSortingExtractor,
]
sorting_extractor_full_list += neo_sorting_extractors_list

event_extractor_full_list = [MaxwellEventExtractor]
event_extractor_full_list += neo_event_extractors_list

snippets_extractor_full_list = [NpySnippetsExtractor, WaveClusSnippetsExtractor]

recording_extractor_full_dict = {}
for rec_class in recording_extractor_full_list:
    # here we get the class name, remove "Recording" and "Extractor" and make it lower case
    rec_class_name = rec_class.__name__.replace("Recording", "").replace("Extractor", "").lower()
    recording_extractor_full_dict[rec_class_name] = rec_class

sorting_extractor_full_dict = {}
for sort_class in sorting_extractor_full_list:
    # here we get the class name, remove "Extractor" and make it lower case
    sort_class_name = sort_class.__name__.replace("Sorting", "").replace("Extractor", "").lower()
    sorting_extractor_full_dict[sort_class_name] = sort_class

event_extractor_full_dict = {}
for event_class in event_extractor_full_list:
    # here we get the class name, remove "Extractor" and make it lower case
    event_class_name = event_class.__name__.replace("Event", "").replace("Extractor", "").lower()
    event_extractor_full_dict[event_class_name] = event_class
