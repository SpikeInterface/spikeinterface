import numpy as np

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo


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
        recgen = self.neo_reader._recgen
        locations = np.array(recgen.channel_positions)
        if locs_2d:
            if 'electrodes' in recgen.info.keys():
                if 'plane' in recgen.info['electrodes'].keys():
                    probe_plane = recgen.info['electrodes']['plane']
                    if probe_plane == 'xy':
                        locations = locations[:, :2]
                    elif probe_plane == 'yz':
                        locations = locations[:, 1:]
                    elif probe_plane == 'xz':
                        locations = locations[:, [0, 2]]
            if locations.shape[1] == 3:
                locations = locations[:, 1:]
        
        print(locations.shape)
        print(self._main_ids)
        self.set_channel_locations(locations)
    
    @staticmethod
    def write_recording(recording, save_path, check_suffix=True):
        # Alessio : I think we don't need this
        raise NotImplementedError


class MEArecSortingExtractor(NeoBaseSortingExtractor):
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'
    handle_raw_spike_directly = True
    
    def __init__(self, file_path, use_natural_unit_ids=True):
        neo_kwargs = {'filename' : file_path}
        NeoBaseSortingExtractor.__init__(self, 
                    sampling_frequency=None, # auto guess is correct here
                    use_natural_unit_ids=use_natural_unit_ids,
                    **neo_kwargs)


    @staticmethod
    def write_sorting(sorting, save_path, sampling_frequency, check_suffix=True):
        # Alessio : I think we don't need this
        raise NotImplementedError
