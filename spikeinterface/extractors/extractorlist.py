# most important extractor are in spikeinterface.core
from spikeinterface.core import BinaryRecordingExtractor, NpzSortingExtractor


from .numpyextractors import NumpyRecording , NumpySorting

# sorting extractors in relation with a sorter
from .klustaextractors import KlustaSortingExtractor
from .hdsortextractors import HDSortSortingExtractor
from .waveclustextractors import WaveClusSortingExtractor


from .neoextractors import (
    MEArecRecordingExtractor, MEArecSortingExtractor,
    SpikeGLXRecordingExtractor,
    OpenEphysLegacyRecordingExtractor, OpenEphysBinaryRecordingExtractor,
    IntanRecordingExtractor,
    NeuroScopeRecordingExtractor,
    PlexonRecordingExtractor,
    NeuralynxRecordingExtractor,
    BlackrockRecordingExtractor,
    MCSRawRecordingExtractor,
    KiloSortSortingExtractor,
    )


recording_extractor_full_list = [
BinaryRecordingExtractor,

    # natively implemented in spikeinterface.extractors
    NumpyRecording,
    
    
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
    
    ##OLD
    #~ MdaRecordingExtractor,
    #~ MEArecRecordingExtractor,          OK
    #~ BiocamRecordingExtractor,
    #~ ExdirRecordingExtractor,    DROP
    #~ OpenEphysRecordingExtractor,  OK
    #~ IntanRecordingExtractor,              OK
    #~ BinDatRecordingExtractor,            OK
    #~ KlustaRecordingExtractor,     
    #~ KiloSortRecordingExtractor,
    #~ SpykingCircusRecordingExtractor,
    #~ SpikeGLXRecordingExtractor,      OK
    #~ PhyRecordingExtractor,
    #~ MaxOneRecordingExtractor,
    #~ Mea1kRecordingExtractor,  DROP
    #~ MCSH5RecordingExtractor, 
    #~ SHYBRIDRecordingExtractor,
    #~ NIXIORecordingExtractor,
    #~ NeuroscopeRecordingExtractor,  OK
    #~ NeuroscopeMultiRecordingTimeExtractor,   OK
    #~ CEDRecordingExtractor,

    #~ # neo based
    #~ PlexonRecordingExtractor,       OK
    #~ NeuralynxRecordingExtractor,  OK
    #~ BlackrockRecordingExtractor,  OK
    #~ MCSRawRecordingExtractor,  OK
]

#~ recording_extractor_dict = {recording_class.extractor_name: recording_class
                            #~ for recording_class in recording_extractor_full_list}
#~ installed_recording_extractor_list = [rx for rx in recording_extractor_full_list if rx.installed]

sorting_extractor_full_list = [
    NpzSortingExtractor,
    
    # natively implemented in spikeinterface.extractors
    NumpySorting,
    KlustaSortingExtractor,
    HDSortSortingExtractor,
    WaveClusSortingExtractor,
    
    # neo based
    MEArecSortingExtractor,
    KiloSortSortingExtractor,
    
    ##OLD
    #~ MdaSortingExtractor,
    #~ MEArecSortingExtractor,   OK
    #~ ExdirSortingExtractor,   DROP
    #~ HDSortSortingExtractor,  
    #~ HS2SortingExtractor,
    #~ KlustaSortingExtractor,  OK
    #~ KiloSortSortingExtractor, OK
    #~ OpenEphysSortingExtractor,
    #~ PhySortingExtractor,        NEAR OK
    #~ SpykingCircusSortingExtractor,   NEAR OK
    #~ TridesclousSortingExtractor,     NEAR OK
    #~ Mea1kSortingExtractor,    DROP
    #~ MaxOneSortingExtractor,
    #~ NpzSortingExtractor,    OK
    #~ SHYBRIDSortingExtractor,
    #~ NIXIOSortingExtractor,
    #~ NeuroscopeSortingExtractor,   DROP ?
    #~ NeuroscopeMultiSortingExtractor,   DROP ?
    #~ WaveClusSortingExtractor,
    #~ YassSortingExtractor,
    #~ CombinatoSortingExtractor,
    #~ ALFSortingExtractor,

    #~ # neo based
    #~ PlexonSortingExtractor,
    #~ NeuralynxSortingExtractor,
    #~ BlackrockSortingExtractor,
    #~ CellExplorerSortingExtractor
]

#~ installed_sorting_extractor_list = [sx for sx in sorting_extractor_full_list if sx.installed]
#~ sorting_extractor_dict = {sorting_class.extractor_name: sorting_class for sorting_class in sorting_extractor_full_list}

#~ writable_sorting_extractor_list = [sx for sx in installed_sorting_extractor_list if sx.is_writable]
#~ writable_sorting_extractor_dict = {sorting_class.extractor_name: sorting_class
                                   #~ for sorting_class in writable_sorting_extractor_list}
