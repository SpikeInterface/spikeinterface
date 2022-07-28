from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Axona RAW format.

    Based on :py:class:`neo.rawio.AxonaRawIO`

    Parameters
    ----------
    folder_path: str
        The folder path to load the recordings from.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'folder'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, file_path, all_annotations=False):
        neo_kwargs = {'filename': file_path}
        NeoBaseRecordingExtractor.__init__(self, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update({'file_path': file_path})


read_axona = define_function_from_class(source_class=AxonaRecordingExtractor, name="read_axona")
