from typing import Type

# most important extractor are in spikeinterface.core
from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BinaryRecordingExtractor, NumpyRecording,
                                 NpzSortingExtractor, NumpySorting,
                                 NpySnippetsExtractor)

# sorting/recording/event from neo
from .neoextractors import *

# non-NEO objects implemented in neo folder
from .neoextractors import NeuroScopeSortingExtractor, MaxwellEventExtractor

# NWB sorting/recording/event
from .nwbextractors import (NwbRecordingExtractor, NwbSortingExtractor,
                            read_nwb, read_nwb_recording, read_nwb_sorting)

from .cbin_ibl import CompressedBinaryIblExtractor, read_cbin_ibl
from .mcsh5extractors import MCSH5RecordingExtractor, read_mcsh5

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

# sorting in relation with simulator
from .shybridextractors import (SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor,
                                read_shybrid_recording, read_shybrid_sorting)

# snippers
from .waveclussnippetstextractors import WaveClusSnippetsExtractor, read_waveclus_snippets


# misc
from .alfsortingextractor import ALFSortingExtractor, read_alf_sorting


########################################

recording_extractor_full_list = [
    BinaryRecordingExtractor,

    # natively implemented in spikeinterface.extractors
    NumpyRecording,
    SHYBRIDRecordingExtractor,
    MdaRecordingExtractor,
    NwbRecordingExtractor,

    # others
    CompressedBinaryIblExtractor,
    MCSH5RecordingExtractor
]
recording_extractor_full_list += neo_recording_extractors_list

sorting_extractor_full_list = [
    NpzSortingExtractor,

    # natively implemented in spikeinterface.extractors
    NumpySorting,
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
]
sorting_extractor_full_list += neo_sorting_extractors_list

event_extractor_full_list = [
    MaxwellEventExtractor
]
event_extractor_full_list += neo_event_extractors_list

snippets_extractor_full_list = [
    NpySnippetsExtractor,
    WaveClusSnippetsExtractor
]


recording_extractor_full_dict = {recext.name: recext for recext in recording_extractor_full_list}
sorting_extractor_full_dict = {recext.name: recext for recext in sorting_extractor_full_list}
snippets_extractor_full_dict = {recext.name: recext for recext in snippets_extractor_full_list}



def get_recording_extractor_from_name(name: str) -> Type[BaseRecording]:
    """
    Returns the Recording Extractor class based on its name.

    Parameters
    ----------
    name: str
        The Recording Extractor's name.

    Returns
    -------
    recording_extractor: BaseRecording
        The Recording Extractor class.
    """

    for recording_extractor in recording_extractor_full_list:
        if recording_extractor.__name__ == name:
            return recording_extractor

    raise ValueError(f"Recording extractor '{name}' not found.")


def get_sorting_extractor_from_name(name: str) -> Type[BaseSorting]:
    """
    Returns the Sorting Extractor class based on its name.

    Parameters
    ----------
    name: str
        The Sorting Extractor's name.

    Returns
    -------
    sorting_extractor: BaseSorting
        The Sorting Extractor class.
    """

    for sorting_extractor in sorting_extractor_full_list:
        if sorting_extractor.__name__ == name:
            return sorting_extractor

    raise ValueError(f"Sorting extractor '{name}' not found.")
