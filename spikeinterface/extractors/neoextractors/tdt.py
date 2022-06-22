from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class TdtRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading TDT folder

    Based on :py:class:`neo.rawio.TdTRawIO`

    Parameters
    ----------
    folder_path: str
        The tdt folder.
    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.

    """
    mode = 'folder'
    NeoRawIOClass = 'TdtRawIO'

    def __init__(self, folder_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'dirname': folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path)))


read_tdt = define_function_from_class(source_class=TdtRecordingExtractor, name="read_tdt")
