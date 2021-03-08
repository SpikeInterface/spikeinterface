import neo

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class IntanRecordingExtractor(NeoBaseRecordingExtractor):
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
    mode = 'file'
    NeoRawIOClass = 'IntanRawIO'
    
    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename' : file_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)