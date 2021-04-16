import neo

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class BlackrockRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading neuralynx folder
    
    Based on neo.rawio.NeuralynxRawIO
    
    Parameters
    ----------
    file_path: str
        The xml  file.
    stream_id: str or None
    """ 
    mode = 'file'
    NeoRawIOClass = 'BlackrockRawIO'

    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename' : str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)
