import numpy as np

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import neo


class SpikeGLXRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a SpikeGLX system  (NI-DAQ for neuropixel probe)
    See https://billkarsh.github.io/SpikeGLX/    
    
    Based on neo.
    
    Parameters
    ----------
    folder_path: str
    """    
    mode = 'file'
    NeoRawIOClass = 'SpikeGLXRawIO'
    
    def __init__(self, folder_path):
        neo_kwargs = {'dirname' : folder_path}
        NeoBaseRecordingExtractor.__init__(self, **neo_kwargs)
