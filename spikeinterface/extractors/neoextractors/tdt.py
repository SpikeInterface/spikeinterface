from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class TdtRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading TDT folder.

    Based on :py:class:`neo.rawio.TdTRawIO`

    Parameters
    ----------
    folder_path: str
        The folder path to the tdt folder.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'folder'
    NeoRawIOClass = 'TdtRawIO'
    name = "tdt"

    def __init__(self, folder_path, stream_id=None, stream_name=None, block_index=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           block_index=block_index,
                                           all_annotations=all_annotations, 
                                           **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path)))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {'dirname': str(folder_path)}
        return neo_kwargs

read_tdt = define_function_from_class(source_class=TdtRecordingExtractor, name="read_tdt")
