from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor
from .neo_utils import get_streams, get_num_blocks


class BlackrockRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading BlackRock data.

    Based on :py:class:`neo.rawio.BlackrockRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'BlackrockRawIO'

    def __init__(self, file_path, stream_id=None, stream_name=None, block_index=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations,
                                           **neo_kwargs)
        self._kwargs.update({'file_path': str(file_path)})


class BlackrockSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading BlackRock spiking data.

    Based on :py:class:`neo.rawio.BlackrockRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    sampling_frequency: float, None by default.
        The sampling frequency for the spiking channels. When the signal data is available (.ncs) those files will be 
        used to extract the frequency. Otherwise, the sampling frequency needs to be specified for this extractor.
    """
    mode = 'file'
    NeoRawIOClass = 'BlackrockRawIO'

    def __init__(self, file_path, sampling_frequency=None):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseSortingExtractor.__init__(self, sampling_frequency=sampling_frequency, **neo_kwargs)
        self._kwargs.update({'file_path': str(file_path)})

read_blackrock = define_function_from_class(source_class=BlackrockRecordingExtractor, name="read_blackrock")
read_blackrock_sorting = define_function_from_class(source_class=BlackrockSortingExtractor, name="read_blackrock_sorting")


def get_blackrock_streams(file_path):
    """Return available NEO streams

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.

    Returns
    -------
    list
        List of stream names
    list
        List of stream IDs
    """
    raw_class = BlackrockRecordingExtractor.NeoRawIOClass
    neo_kwargs = {'filename': str(file_path)}
    return get_streams(raw_class, **neo_kwargs)
