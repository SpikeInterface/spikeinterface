from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor
from .neo_utils import get_streams, get_num_blocks


class CedRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading smr/smrw CED file.

    Based on :py:class:`neo.rawio.CedRawIO` / sonpy

    Alternative to read_spike2 which does not handle smrx

    Parameters
    ----------
    file_path: str
        The file path to the smr or smrx file.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    block_index: int, optional
        If there are several blocks, specify the block index you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'CedRawIO'

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations,
                                           **neo_kwargs)
        self._kwargs.update(dict(file_path=str(file_path)))
        self.extra_requirements.append('sonpy')



read_ced = define_function_from_class(source_class=CedRecordingExtractor, name="read_ced")


def get_ced_streams(file_path):
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
    raw_class = CedRecordingExtractor.NeoRawIOClass
    neo_kwargs = {'filename': str(file_path)}
    return get_streams(raw_class, **neo_kwargs)
