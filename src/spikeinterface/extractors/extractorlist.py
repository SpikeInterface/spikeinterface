from __future__ import annotations


# most important extractor are in spikeinterface.core
from spikeinterface.core import (
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
    read_npy_snippets,
)

# sorting/recording/event from neo
from .neoextractors import *

# non-NEO objects implemented in neo folder
# keep for reference Currently pulling from neoextractor __init__
# from .neoextractors import NeuroScopeSortingExtractor, MaxwellEventExtractor

# NWB sorting/recording/event
from .nwbextractors import (
    NwbRecordingExtractor,
    NwbSortingExtractor,
    NwbTimeSeriesExtractor,  # @Heberto/Alessio what do we do with this
    read_nwb,
    read_nwb_recording,
    read_nwb_sorting,
    read_nwb_timeseries,
)

from .cbin_ibl import CompressedBinaryIblExtractor, read_cbin_ibl
from .iblextractors import IblRecordingExtractor, IblSortingExtractor, read_ibl_recording, read_ibl_sorting
from .mcsh5extractors import MCSH5RecordingExtractor, read_mcsh5
from .whitematterrecordingextractor import WhiteMatterRecordingExtractor, read_whitematter

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


#################################################
# first we line up each class with its function

_recording_extractor_full_dict = {
    BinaryFolderRecording: read_binary,
    BinaryRecordingExtractor: read_binary,
    ZarrRecordingExtractor: read_zarr,
    # natively implemented in spikeinterface.extractors
    NumpyRecording: NumpyRecording,
    SHYBRIDRecordingExtractor: read_shybrid_recording,
    MdaRecordingExtractor: read_mda_recording,
    NwbRecordingExtractor: read_nwb_recording,
    # others
    CompressedBinaryIblExtractor: read_cbin_ibl,
    IblRecordingExtractor: read_ibl_recording,
    MCSH5RecordingExtractor: read_mcsh5,
    SinapsResearchPlatformRecordingExtractor: read_sinaps_research_platform,
    SinapsResearchPlatformH5RecordingExtractor: read_sinaps_research_platform_h5,
    WhiteMatterRecordingExtractor: read_whitematter,
}
_recording_extractor_full_dict.update(neo_recording_extractors_dict)

_sorting_extractor_full_dict = {
    NpzSortingExtractor: read_npz_sorting,
    ZarrSortingExtractor: read_zarr,
    NumpySorting: NumpySorting,
    # natively implemented in spikeinterface.extractors
    MdaSortingExtractor: read_mda_sorting,
    SHYBRIDSortingExtractor: read_shybrid_sorting,
    ALFSortingExtractor: read_alf_sorting,
    KlustaSortingExtractor: read_klusta,
    HDSortSortingExtractor: read_hdsort,
    MClustSortingExtractor: read_mclust,
    WaveClusSortingExtractor: read_waveclus,
    YassSortingExtractor: read_yass,
    CombinatoSortingExtractor: read_combinato,
    TridesclousSortingExtractor: read_tridesclous,
    SpykingCircusSortingExtractor: read_spykingcircus,
    HerdingspikesSortingExtractor: read_herdingspikes,
    KiloSortSortingExtractor: read_kilosort,
    PhySortingExtractor: read_phy,
    NwbSortingExtractor: read_nwb_sorting,
    IblSortingExtractor: read_ibl_sorting,
    CellExplorerSortingExtractor: read_cellexplorer,
}
_sorting_extractor_full_dict.update(neo_sorting_extractors_dict)

# events only from neo
_event_extractor_full_dict = neo_event_extractors_dict

_snippets_extractor_full_dict = {
    NpySnippetsExtractor: read_npy_snippets,
    WaveClusSnippetsExtractor: read_waveclus_snippets,
}

#############################################################################################
# Organize the possible extractors into an easy to use format

recording_extractor_full_dict = {
    rec_class.__name__.replace("Recording", "").replace("Extractor", "").lower(): rec_func.__name__
    for rec_class, rec_func in _recording_extractor_full_dict.items()
}
sorting_extractor_full_dict = {
    sort_class.__name__.replace("Sorting", "").replace("Extractor", "").lower(): sort_func.__name__
    for sort_class, sort_func in _sorting_extractor_full_dict.items()
}
event_extractor_full_dict = {
    event_class.__name__.replace("Event", "").replace("Extractor", "").lower(): event_func.__name__
    for event_class, event_func in _event_extractor_full_dict.items()
}
snippets_extractor_full_dict = {
    snippets_class.__name__.replace("Snippets", "").replace("Extractor", "").lower(): snippets_func.__name__
    for snippets_class, snippets_func in _snippets_extractor_full_dict.items()
}


# we only do the functions in the init rather than pull in the classes
__all__ = list(recording_extractor_full_dict.values())
__all__ += list(sorting_extractor_full_dict.values())
__all__ += list(event_extractor_full_dict.values())
__all__ += list(snippets_extractor_full_dict.values())
__all__.extend(
    [
        "read_nwb",
        "recording_extractor_full_dict",
        "sorting_extractor_full_dict",
        "event_extractor_full_dict",
        "snippets_extractor_full_dict",
        "read_phy",
    ]
)
