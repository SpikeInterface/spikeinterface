from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading neuralynx folder

    Based on :py:class:`neo.rawio.NeuralynxRawIO`

    Parameters
    ----------
    folder_path: str
        The file path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'folder'
    NeoRawIOClass = 'NeuralynxRawIO'
    name = "neuralynx"

    def __init__(self, folder_path, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations, 
                                           **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path)))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {'dirname': str(folder_path)}
        return neo_kwargs


class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading spike data from a folder with neuralynx spiking data (i.e .nse and .ntt formats).

    Based on :py:class:`neo.rawio.NeuralynxRawIO`

    Parameters
    ----------
    folder_path: str
        The file path to load the recordings from.
    sampling_frequency: float
        The sampling frequency for the spiking channels. When the signal data is available (.ncs) those files will be 
        used to extract the frequency. Otherwise, the sampling frequency needs to be specified for this extractor.
    """
    mode = 'folder'
    NeoRawIOClass = 'NeuralynxRawIO'
    handle_spike_frame_directly = False
    name = "neuralynx"

    def __init__(self, folder_path, sampling_frequency=None):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseSortingExtractor.__init__(self, sampling_frequency=sampling_frequency, **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path)))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {'dirname': str(folder_path)}
        return neo_kwargs


read_neuralynx = define_function_from_class(source_class=NeuralynxRecordingExtractor, name="read_neuralynx")
read_neuralynx_sorting = define_function_from_class(source_class=NeuralynxSortingExtractor,
                                                    name="read_neuralynx_sorting")
