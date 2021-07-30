# most important extractor are in spikeinterface.core
from spikeinterface.core import (BinaryRecordingExtractor,
                                 NpzSortingExtractor, NumpyRecording, NumpySorting)

# sorting/recording/event from neo
from .neoextractors import (
    MEArecRecordingExtractor, MEArecSortingExtractor, read_mearec,
    SpikeGLXRecordingExtractor, read_spikeglx,
    OpenEphysLegacyRecordingExtractor, OpenEphysBinaryRecordingExtractor, OpenEphysBinaryEventExtractor, read_openephys,
    IntanRecordingExtractor, read_intan,
    NeuroScopeRecordingExtractor, read_neuroscope,
    PlexonRecordingExtractor, read_plexon,
    NeuralynxRecordingExtractor, read_neuralynx,
    BlackrockRecordingExtractor, read_blackrock,
    MCSRawRecordingExtractor, read_mcsraw,
    Spike2RecordingExtractor, read_spike2,
    CedRecordingExtractor, read_ced,
    MaxwellRecordingExtractor, read_maxwell,
    NixRecordingExtractor, read_nix,
    SpikeGadgetsRecordingExtractor, read_spikegadgets,

)

# NWB sorting/recording/event
from .nwbextractors import (NwbRecordingExtractor, NwbSortingExtractor,
                            read_nwb, read_nwb_recording, read_nwb_sorting)

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

    # neo based
    MEArecSortingExtractor
]

event_extractor_full_list = [
    OpenEphysBinaryEventExtractor,
]
