import numpy as np

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo

from probeinterface import read_mearec

class MEArecRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a MEArec simulated data.
    
    
    Parameters
    ----------
    file_path: str
    
    locs_2d: bool
    
    """    
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'
    
    def __init__(self, file_path, locs_2d=True):
        
        neo_kwargs = {'filename' : file_path}
        NeoBaseRecordingExtractor.__init__(self, **neo_kwargs)
        
        # channel location 
        # this will be handled with probeinterface (when Alessio will do his homework)
        #~ recgen = self.neo_reader._recgen
        #~ locations = np.array(recgen.channel_positions)
        #~ if locs_2d:
            #~ if 'electrodes' in recgen.info.keys():
                #~ if 'plane' in recgen.info['electrodes'].keys():
                    #~ probe_plane = recgen.info['electrodes']['plane']
                    #~ if probe_plane == 'xy':
                        #~ locations = locations[:, :2]
                    #~ elif probe_plane == 'yz':
                        #~ locations = locations[:, 1:]
                    #~ elif probe_plane == 'xz':
                        #~ locations = locations[:, [0, 2]]
            #~ if locations.shape[1] == 3:
                #~ locations = locations[:, 1:]
        #~ self.set_channel_locations(locations)
        
        probe = read_mearec(file_path)
        self.set_probe(probe, in_place=True)
        
        self._kwargs = {'file_path' : str(file_path)}
        
    

class MEArecSortingExtractor(NeoBaseSortingExtractor):
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'
    handle_spike_frame_directly = False
    
    def __init__(self, file_path, use_natural_unit_ids=True):
        neo_kwargs = {'filename' : file_path}
        NeoBaseSortingExtractor.__init__(self, 
                    sampling_frequency=None, # auto guess is correct here
                    use_natural_unit_ids=use_natural_unit_ids,
                    **neo_kwargs)
        
        self._kwargs = {'file_path' : str(file_path), 'use_natural_unit_ids': use_natural_unit_ids}
