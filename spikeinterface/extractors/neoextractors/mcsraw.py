from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class MCSRawRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from "Raw" Multi Channel System (MCS) format.
    This format is NOT the native MCS format (*.mcd).
    This format is a raw format with an internal binary header exported by the
    "MC_DataTool binary conversion" with the option header selected.

    Based on :py:class:`neo.rawio.RawMCSRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
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
    NeoRawIOClass = 'RawMCSRawIO'
    name = "mcsraw"

    def __init__(self, file_path, stream_id=None, stream_name=None, block_index=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id,
                                           stream_name=stream_name,
                                           block_index=block_index,
                                           all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update(dict(file_path=str(file_path)))

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {'filename': str(file_path)}
        return neo_kwargs

read_mcsraw = define_function_from_class(source_class=MCSRawRecordingExtractor, name="read_maxwell_event")
