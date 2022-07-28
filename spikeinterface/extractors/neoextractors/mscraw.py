from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


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
        If there are several streams, specify the one you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'RawMCSRawIO'

    def __init__(self, file_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'filename': file_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update(dict(file_path=str(file_path)))


read_mcsraw = define_function_from_class(source_class=MCSRawRecordingExtractor, name="read_maxwell_event")
