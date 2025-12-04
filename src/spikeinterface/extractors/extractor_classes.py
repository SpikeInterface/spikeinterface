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
from .neoextractors import read_neuroscope
from .neoextractors.intan import read_split_intan_files, IntanSplitFilesRecordingExtractor

# non-NEO objects implemented in neo folder
# keep for reference Currently pulling from neoextractor __init__
# from .neoextractors import NeuroScopeSortingExtractor, MaxwellEventExtractor

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


###############################################################################################
# the following code is necessary for controlling what the end user imports from spikeinterface.
# The strategy has three goals:
#
#    * A mapping from the original class to its wrapper (because that's what we want to expose)
#    * A mapping from the original class to its wrapper string (because of __all__)
#    * A mapping from format to the class wrapper for convenience (exposed to users for ease of use)
#
# To achieve these three goals we do the following:
#
# 1) we line up each class with its wrapper that returns a snakecase version of the class (in some docs called
#    the "function" version, although this is just a wrapper of the underlying class)
# 2) we do (1) by creating nested dicts where the key is the original class and the values are a nested dict with
# 3) a "wrapper_class" key which returns the wrapper to be exposed to the end user and
# 4) a "wrapper_string" which is added to the __all__ attribute of the __init__. This is necessary because __all__
#    can only accept a list of strings
# 5) Finally we create dictionaries exposed to the user where we return a formatted file format as a key along
#    with the value being the wrapper (see the comment below for examples for this dict)
#
# Note that some formats (e.g. binary and numpy) still use the class format as they aren't read-only (i.e. they
# have no wrapper)

_recording_extractor_full_dict = {
    # core extractors that are returned as classes
    BinaryFolderRecording: dict(wrapper_string="BinaryFolderRecording", wrapper_class=BinaryFolderRecording),
    BinaryRecordingExtractor: dict(wrapper_string="BinaryRecordingExtractor", wrapper_class=BinaryRecordingExtractor),
    ZarrRecordingExtractor: dict(wrapper_string="ZarrRecordingExtractor", wrapper_class=ZarrRecordingExtractor),
    # natively implemented in spikeinterface.extractors
    NumpyRecording: dict(wrapper_string="NumpyRecording", wrapper_class=NumpyRecording),
    SHYBRIDRecordingExtractor: dict(wrapper_string="read_shybrid_recording", wrapper_class=read_shybrid_recording),
    MdaRecordingExtractor: dict(wrapper_string="read_mda_recording", wrapper_class=read_mda_recording),
    NwbRecordingExtractor: dict(wrapper_string="read_nwb_recording", wrapper_class=read_nwb_recording),
    NwbTimeSeriesExtractor: dict(wrapper_string="read_nwb_timeseries", wrapper_class=read_nwb_timeseries),
    # others
    CompressedBinaryIblExtractor: dict(wrapper_string="read_cbin_ibl", wrapper_class=read_cbin_ibl),
    IblRecordingExtractor: dict(wrapper_string="read_ibl_recording", wrapper_class=read_ibl_recording),
    MCSH5RecordingExtractor: dict(wrapper_string="read_mcsh5", wrapper_class=read_mcsh5),
    SinapsResearchPlatformRecordingExtractor: dict(
        wrapper_string="read_sinaps_research_platform", wrapper_class=read_sinaps_research_platform
    ),
    SinapsResearchPlatformH5RecordingExtractor: dict(
        wrapper_string="read_sinaps_research_platform_h5", wrapper_class=read_sinaps_research_platform_h5
    ),
    WhiteMatterRecordingExtractor: dict(wrapper_string="read_whitematter", wrapper_class=read_whitematter),
}
_recording_extractor_full_dict.update(neo_recording_extractors_dict)

_sorting_extractor_full_dict = {
    NpzSortingExtractor: dict(wrapper_string="read_npz_sorting", wrapper_class=read_npz_sorting),
    ZarrSortingExtractor: dict(wrapper_string="ZarrSortingExtractor", wrapper_class=ZarrSortingExtractor),
    NumpySorting: dict(wrapper_string="NumpySorting", wrapper_class=NumpySorting),
    # natively implemented in spikeinterface.extractors
    MdaSortingExtractor: dict(wrapper_string="read_mda_sorting", wrapper_class=read_mda_sorting),
    SHYBRIDSortingExtractor: dict(wrapper_string="read_shybrid_sorting", wrapper_class=read_shybrid_sorting),
    ALFSortingExtractor: dict(wrapper_string="read_alf_sorting", wrapper_class=read_alf_sorting),
    KlustaSortingExtractor: dict(wrapper_string="read_klusta", wrapper_class=read_klusta),
    HDSortSortingExtractor: dict(wrapper_string="read_hdsort", wrapper_class=read_hdsort),
    MClustSortingExtractor: dict(wrapper_string="read_mclust", wrapper_class=read_mclust),
    WaveClusSortingExtractor: dict(wrapper_string="read_waveclus", wrapper_class=read_waveclus),
    YassSortingExtractor: dict(wrapper_string="read_yass", wrapper_class=read_yass),
    CombinatoSortingExtractor: dict(wrapper_string="read_combinato", wrapper_class=read_combinato),
    TridesclousSortingExtractor: dict(wrapper_string="read_tridesclous", wrapper_class=read_tridesclous),
    SpykingCircusSortingExtractor: dict(wrapper_string="read_spykingcircus", wrapper_class=read_spykingcircus),
    HerdingspikesSortingExtractor: dict(wrapper_string="read_herdingspikes", wrapper_class=read_herdingspikes),
    KiloSortSortingExtractor: dict(wrapper_string="read_kilosort", wrapper_class=read_kilosort),
    PhySortingExtractor: dict(wrapper_string="read_phy", wrapper_class=read_phy),
    NwbSortingExtractor: dict(wrapper_string="read_nwb_sorting", wrapper_class=read_nwb_sorting),
    IblSortingExtractor: dict(wrapper_string="read_ibl_sorting", wrapper_class=read_ibl_sorting),
    CellExplorerSortingExtractor: dict(wrapper_string="read_cellexplorer", wrapper_class=read_cellexplorer),
}
_sorting_extractor_full_dict.update(neo_sorting_extractors_dict)

# events only from neo
_event_extractor_full_dict = neo_event_extractors_dict

_snippets_extractor_full_dict = {
    NpySnippetsExtractor: dict(wrapper_string="read_npy_snippets", wrapper_class=read_npy_snippets),
    WaveClusSnippetsExtractor: dict(wrapper_string="read_waveclus_snippets", wrapper_class=read_waveclus_snippets),
}

############################################################################################################
# Organize the possible extractors into a user facing format with keys being extractor names
# (e.g. 'intan' , 'kilosort') and values being the appropriate Extractor class returned as its wrapper
# (e.g. IntanRecordingExtractor, KiloSortSortingExtractor)
# An important note is the the formats are returned after performing `.lower()` so a format like
# SpikeGLX will have a key of 'spikeglx'
# for example if we wanted to create a recording from an intan file we could do the following:
# >>> recording = se.recording_extractor_full_dict['intan'](file_path='path/to/data.rhd')


recording_extractor_full_dict = {
    rec_class.__name__.replace("Recording", "").replace("Extractor", "").lower(): rec_func["wrapper_class"]
    for rec_class, rec_func in _recording_extractor_full_dict.items()
}
sorting_extractor_full_dict = {
    sort_class.__name__.replace("Sorting", "").replace("Extractor", "").lower(): sort_func["wrapper_class"]
    for sort_class, sort_func in _sorting_extractor_full_dict.items()
}
event_extractor_full_dict = {
    event_class.__name__.replace("Event", "").replace("Extractor", "").lower(): event_func["wrapper_class"]
    for event_class, event_func in _event_extractor_full_dict.items()
}
snippets_extractor_full_dict = {
    snippets_class.__name__.replace("Snippets", "").replace("Extractor", "").lower(): snippets_func["wrapper_class"]
    for snippets_class, snippets_func in _snippets_extractor_full_dict.items()
}


# we only do the functions in the init rather than pull in the classes
__all__ = [func["wrapper_string"] for func in _recording_extractor_full_dict.values()]
__all__ += [func["wrapper_string"] for func in _sorting_extractor_full_dict.values()]
__all__ += [func["wrapper_string"] for func in _event_extractor_full_dict.values()]
__all__ += [func["wrapper_string"] for func in _snippets_extractor_full_dict.values()]
__all__.extend(
    [
        "read_nwb",  # convenience function for multiple nwb formats
        "recording_extractor_full_dict",
        "sorting_extractor_full_dict",
        "event_extractor_full_dict",
        "snippets_extractor_full_dict",
        "read_binary",  # convenience function for binary formats
        "read_zarr",
        "read_neuroscope",  # convenience function for neuroscope
        "read_split_intan_files",  # convenience function for segmented intan files
    ]
)
