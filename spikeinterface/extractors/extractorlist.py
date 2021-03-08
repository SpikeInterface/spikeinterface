# most important extractor are in spikeinterface.core
from spikeinterface.core import BinaryRecordingExtractor, NpzSortingExtractor


from .numpyextractors import NumpyRecording , NumpySorting

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
    #~ ExdirRecordingExtractor,
    #~ OpenEphysRecordingExtractor,  OK
    #~ IntanRecordingExtractor,              OK
    #~ BinDatRecordingExtractor,            OK
    #~ KlustaRecordingExtractor,
    #~ KiloSortRecordingExtractor,
    #~ SpykingCircusRecordingExtractor,
    #~ SpikeGLXRecordingExtractor,      OK
    #~ PhyRecordingExtractor,
    #~ MaxOneRecordingExtractor,
    #~ Mea1kRecordingExtractor,
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
    
    # neo based
    MEArecSortingExtractor,
    
    ##OLD
    #~ MdaSortingExtractor,
    #~ MEArecSortingExtractor,
    #~ ExdirSortingExtractor,
    #~ HDSortSortingExtractor,
    #~ HS2SortingExtractor,
    #~ KlustaSortingExtractor,
    #~ KiloSortSortingExtractor,
    #~ OpenEphysSortingExtractor,
    #~ PhySortingExtractor,
    #~ SpykingCircusSortingExtractor,
    #~ TridesclousSortingExtractor,
    #~ Mea1kSortingExtractor,
    #~ MaxOneSortingExtractor,
    #~ NpzSortingExtractor,
    #~ SHYBRIDSortingExtractor,
    #~ NIXIOSortingExtractor,
    #~ NeuroscopeSortingExtractor,
    #~ NeuroscopeMultiSortingExtractor,
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
