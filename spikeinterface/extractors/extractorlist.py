# most important extractor are in spikeinterface.core
from spikeinterface.core import (BinaryRecordingExtractor,
                                 NpzSortingExtractor, NumpyRecording, NumpySorting)

# sorting/recording/event from neo
from .neoextractors import (
    MEArecRecordingExtractor, MEArecSortingExtractor, read_mearec,
    SpikeGLXRecordingExtractor, read_spikeglx,
    OpenEphysLegacyRecordingExtractor, OpenEphysBinaryRecordingExtractor, OpenEphysBinaryEventExtractor, 
    read_openephys, read_openephys_event,
    IntanRecordingExtractor, read_intan,
    NeuroScopeRecordingExtractor, read_neuroscope_recording, 
    NeuroScopeSortingExtractor, read_neuroscope_sorting,
    read_neuroscope,
    PlexonRecordingExtractor, read_plexon,
    NeuralynxRecordingExtractor, read_neuralynx,
    BlackrockRecordingExtractor, read_blackrock,
    MCSRawRecordingExtractor, read_mcsraw,
    Spike2RecordingExtractor, read_spike2,
    CedRecordingExtractor, read_ced,
    MaxwellRecordingExtractor, read_maxwell, MaxwellEventExtractor, read_maxwell_event,
    NixRecordingExtractor, read_nix,
    SpikeGadgetsRecordingExtractor, read_spikegadgets,
    BiocamRecordingExtractor, read_biocam,
    AxonaRecordingExtractor, read_axona,
    TdtRecordingExtractor, read_tdt,
    AlphaOmegaRecordingExtractor, read_alphaomega,
    AlphaOmegaEventExtractor, read_alphaomega_event,
)

# NWB sorting/recording/event
from .nwbextractors import (NwbRecordingExtractor, NwbSortingExtractor,
                            read_nwb, read_nwb_recording, read_nwb_sorting)

from .cbin_ibl import CompressedBinaryIblExtractor, read_cbin_ibl
from .mcsh5extractors import MCSH5RecordingExtractor, read_mcsh5

# sorting extractors in relation with a sorter
from .klustaextractors import KlustaSortingExtractor, read_klusta
from .hdsortextractors import HDSortSortingExtractor, read_hdsort
from .waveclustextractors import WaveClusSortingExtractor, read_waveclust
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

# misc
from .alfsortingextractor import ALFSortingExtractor, read_alf_sorting
from .neuropixels_utils import get_neuropixels_channel_groups, get_neuropixels_sample_shifts

#snippets 
from .numpysnippetsextractor import NumpySnippetsExtractor

########################################

recording_extractor_full_list = [
    BinaryRecordingExtractor,

    # natively implemented in spikeinterface.extractors
    NumpyRecording,
    SHYBRIDRecordingExtractor,
    MdaRecordingExtractor,

    # neo based
    MEArecRecordingExtractor,
    SpikeGLXRecordingExtractor,
    OpenEphysLegacyRecordingExtractor,
    OpenEphysBinaryRecordingExtractor,
    IntanRecordingExtractor,
    NeuroScopeRecordingExtractor,
    PlexonRecordingExtractor,
    NeuralynxRecordingExtractor,
    BlackrockRecordingExtractor,
    MCSRawRecordingExtractor,
    Spike2RecordingExtractor,
    CedRecordingExtractor,
    MaxwellRecordingExtractor,
    NixRecordingExtractor,
    NwbRecordingExtractor,
    SpikeGadgetsRecordingExtractor,
    BiocamRecordingExtractor,
    AxonaRecordingExtractor,
    TdtRecordingExtractor,
    AlphaOmegaRecordingExtractor,

    # others
    CompressedBinaryIblExtractor,
    MCSH5RecordingExtractor
]

sorting_extractor_full_list = [
    NpzSortingExtractor,

    # natively implemented in spikeinterface.extractors
    NumpySorting,
    MdaSortingExtractor,
    SHYBRIDSortingExtractor,
    ALFSortingExtractor,

    KlustaSortingExtractor,
    HDSortSortingExtractor,
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

    # neo based
    MEArecSortingExtractor
]

event_extractor_full_list = [
    OpenEphysBinaryEventExtractor,
    MaxwellEventExtractor,
    AlphaOmegaEventExtractor
]
