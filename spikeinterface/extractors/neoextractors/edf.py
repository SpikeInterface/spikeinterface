from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor
from .neo_utils import get_streams, get_num_blocks


class EDFRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading EDF (European data format) folder.

    Based on :py:class:`neo.rawio.EDFRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
        For this neo reader streams are defined by their sampling frequency.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    block_index: int, optional
        If there are several blocks, specify the block index you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'EDFRawIO'

    def __init__(self, file_path, stream_id=None, stream_name=None, block_index=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           block_index=block_index,
                                           all_annotations=all_annotations,
                                           **neo_kwargs)
        self._kwargs.update({'file_path': str(file_path)})
        self.extra_requirements.append('pyedflib')


read_edf = define_function_from_class(source_class=EDFRecordingExtractor, name="read_edf")


def get_edf_streams(file_path):
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
    raw_class = EDFRecordingExtractor.NeoRawIOClass
    neo_kwargs = {'filename': str(file_path)}
    return get_streams(raw_class, **neo_kwargs)


def get_edf_num_blocks(file_path):
    """Return number of NEO blocks

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.

    Returns
    -------
    int
        Number of NEO blocks
    """
    raw_class = EDFRecordingExtractor.NeoRawIOClass
    neo_kwargs = {'filename': str(file_path)}
    return get_num_blocks(raw_class, **neo_kwargs)
