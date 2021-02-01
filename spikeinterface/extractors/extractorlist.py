from .numpyextractors import NumpyRecording , NumpySorting

from .neoextractors import(
    MEArecRecordingExtractor, MEArecSortingExtractor,
    SpikeGLXRecordingExtractor,
    )


recording_extractor_full_list = [
    NumpyRecording,
    
    # neo based
    MEArecRecordingExtractor,
    SpikeGLXRecordingExtractor,
    
    ##OLD
    #~ MdaRecordingExtractor,
    #~ MEArecRecordingExtractor,
    #~ BiocamRecordingExtractor,
    #~ ExdirRecordingExtractor,
    #~ OpenEphysRecordingExtractor,
    #~ IntanRecordingExtractor,
    #~ BinDatRecordingExtractor,
    #~ KlustaRecordingExtractor,
    #~ KiloSortRecordingExtractor,
    #~ SpykingCircusRecordingExtractor,
    #~ SpikeGLXRecordingExtractor,
    #~ PhyRecordingExtractor,
    #~ MaxOneRecordingExtractor,
    #~ Mea1kRecordingExtractor,
    #~ MCSH5RecordingExtractor,
    #~ SHYBRIDRecordingExtractor,
    #~ NIXIORecordingExtractor,
    #~ NeuroscopeRecordingExtractor,
    #~ NeuroscopeMultiRecordingTimeExtractor,
    #~ CEDRecordingExtractor,

    #~ # neo based
    #~ PlexonRecordingExtractor,
    #~ NeuralynxRecordingExtractor,
    #~ BlackrockRecordingExtractor,
    #~ MCSRawRecordingExtractor,
]

#~ recording_extractor_dict = {recording_class.extractor_name: recording_class
                            #~ for recording_class in recording_extractor_full_list}
#~ installed_recording_extractor_list = [rx for rx in recording_extractor_full_list if rx.installed]

sorting_extractor_full_list = [
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
