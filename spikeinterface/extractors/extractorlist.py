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
    KiloSortSortingExtractor, read_kilosort,
    Spike2RecordingExtractor, read_spike2,
    
    # not in python-neo master yet
    # MaxwellRecordingExtractor,  
    # CedRecordingExtractor,  
)

# NWB sorting/recording/event
from .nwbextractors import (NwbRecordingExtractor, NwbSortingExtractor
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

# sorting in relation with simulator
from .shybridextractors import (  SHYBRIDRecordingExtractor, SHYBRIDSortingExtractor,
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
    
    # not in python-neo master yet
    # MaxwellRecordingExtractor,

    ##OLD
    # ~ MdaRecordingExtractor,
    # ~ MEArecRecordingExtractor,          OK
    # ~ BiocamRecordingExtractor,
    # ~ ExdirRecordingExtractor,    DROP
    # ~ OpenEphysRecordingExtractor,  OK
    # ~ IntanRecordingExtractor,              OK
    # ~ BinDatRecordingExtractor,            OK
    # ~ KlustaRecordingExtractor,
    # ~ KiloSortRecordingExtractor,
    # ~ SpykingCircusRecordingExtractor,
    # ~ SpikeGLXRecordingExtractor,      OK
    # ~ PhyRecordingExtractor,
    # ~ MaxOneRecordingExtractor,
    # ~ Mea1kRecordingExtractor,  DROP
    # ~ MCSH5RecordingExtractor,
    # ~ SHYBRIDRecordingExtractor,
    # ~ NIXIORecordingExtractor,
    # ~ NeuroscopeRecordingExtractor,  OK
    # ~ NeuroscopeMultiRecordingTimeExtractor,   OK
    # ~ CEDRecordingExtractor,

    # ~ # neo based
    # ~ PlexonRecordingExtractor,       OK
    # ~ NeuralynxRecordingExtractor,  OK
    # ~ BlackrockRecordingExtractor,  OK
    # ~ MCSRawRecordingExtractor,  OK
]

# ~ recording_extractor_dict = {recording_class.extractor_name: recording_class
# ~ for recording_class in recording_extractor_full_list}
# ~ installed_recording_extractor_list = [rx for rx in recording_extractor_full_list if rx.installed]

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

    # neo based
    MEArecSortingExtractor,
    KiloSortSortingExtractor,

    ##OLD
    # ~ MdaSortingExtractor,
    # ~ MEArecSortingExtractor,   OK
    # ~ ExdirSortingExtractor,   DROP
    # ~ HDSortSortingExtractor,
    # ~ HS2SortingExtractor,
    # ~ KlustaSortingExtractor,  OK
    # ~ KiloSortSortingExtractor, OK
    # ~ OpenEphysSortingExtractor,
    # ~ PhySortingExtractor,        NEAR OK
    # ~ SpykingCircusSortingExtractor,   NEAR OK
    # ~ TridesclousSortingExtractor,     NEAR OK
    # ~ Mea1kSortingExtractor,    DROP
    # ~ MaxOneSortingExtractor,
    # ~ NpzSortingExtractor,    OK
    # ~ SHYBRIDSortingExtractor,
    # ~ NIXIOSortingExtractor,
    # ~ NeuroscopeSortingExtractor,   DROP ?
    # ~ NeuroscopeMultiSortingExtractor,   DROP ?
    # ~ WaveClusSortingExtractor,     OK
    # ~ YassSortingExtractor,                OK
    # ~ CombinatoSortingExtractor,   OK
    # ~ ALFSortingExtractor,

    # ~ # neo based
    # ~ PlexonSortingExtractor,
    # ~ NeuralynxSortingExtractor,
    # ~ BlackrockSortingExtractor,
    # ~ CellExplorerSortingExtractor
]

event_extractor_full_list = [
    OpenEphysBinaryEventExtractor,
]

# ~ installed_sorting_extractor_list = [sx for sx in sorting_extractor_full_list if sx.installed]
# ~ sorting_extractor_dict = {sorting_class.extractor_name: sorting_class for sorting_class in sorting_extractor_full_list}

# ~ writable_sorting_extractor_list = [sx for sx in installed_sorting_extractor_list if sx.is_writable]
# ~ writable_sorting_extractor_dict = {sorting_class.extractor_name: sorting_class
# ~ for sorting_class in writable_sorting_extractor_list}
