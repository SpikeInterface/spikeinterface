from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading neuralynx folder
    
    Based on neo.rawio.NeuralynxRawIO
    
    Parameters
    ----------
    folder_path: str
        The xml  file.
    stream_id: str or None
    """
    mode = 'folder'
    NeoRawIOClass = 'NeuralynxRawIO'

    def __init__(self, folder_path, stream_id=None):
        neo_kwargs = {'dirname': folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id)


read_neuralynx = define_function_from_class(source_class=NeuralynxRecordingExtractor, name="read_neuralynx")
