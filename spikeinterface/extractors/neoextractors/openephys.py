"""
There are 2 openephys reader:
  * OpenEphysLegacyRecordingExtractor: old one aka "open ephys format"
  * OpenEphysBinaryRecordingExtractor: new one aka "binary format"

https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/index.html
"""
import neo
from pathlib import Path

from .neobaseextractor import (NeoBaseRecordingExtractor,
    NeoBaseSortingExtractor, 
    NeoBaseEventExtractor)


class OpenEphysLegacyRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a OpenEphy board.
    
    This open the openephys "legacy" format: one file per channel.
    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Open-Ephys-format.html
    
    Based on neo.rawio.OpenEphysRawIO
    
    Parameters
    ----------
    folder_path: str
        
    stream_id: str or None
        stream for instance : 'imec0.ap' 'nidq' or 'imec0.lf'
    """ 
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysRawIO'
    
    def __init__(self, folder_path, stream_id=None):
        neo_kwargs = {'dirname' : folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)
        
        self._kwargs = dict(folder_path=folder_path, stream_id=stream_id)


class OpenEphysBinaryRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading traces from a OpenEphy board.
    
    This open the openephys "new" "binary" format: one file per continuous stream.
    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html
    
    Based on neo.rawio.OpenEphysBinaryRawIO
    
    Parameters
    ----------
    folder_path: str
        
    stream_id: str or None
        stream for instance : 'imec0.ap' 'nidq' or 'imec0.lf'
    """ 
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysBinaryRawIO'
    
    def __init__(self, folder_path, stream_id=None):
        neo_kwargs = {'dirname' : folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)
        
        self._kwargs = dict(folder_path=folder_path, stream_id=stream_id)



class OpenEphysBinaryEventExtractor(NeoBaseEventExtractor):
    """
    Class for reading events from a OpenEphy board.
    
    This open the openephys "new" "binary" format: one file per continuous stream.
    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html
    
    Based on neo.rawio.OpenEphysBinaryRawIO
    
    Parameters
    ----------
    folder_path: str
        
    """ 
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysBinaryRawIO'
    
    def __init__(self, folder_path):
        neo_kwargs = {'dirname' : folder_path}
        NeoBaseEventExtractor.__init__(self, **neo_kwargs)


def read_openephys(folder_path, **kwargs):
    # auto guess format
    files = [str(f) for f in Path(folder_path).iterdir()]
    if np.any([f.startswith('Continuous') for f in files]):
        # format = 'legacy'
        recording = OpenEphysLegacyRecordingExtractor(folder_path,  **kwargs)
    else:
        # format = 'binary'
        recording = OpenEphysBinaryRecordingExtractor(folder_path,  **kwargs)
    return recording
    
    


