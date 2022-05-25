from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Axona RAW format

    Based on neo.rawio.AxonaRawIO

    Parameters
    ----------
    folder_path: str
        The tdt folder.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
    """
    mode = 'folder'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, file_path, all_annotations=False):
        neo_kwargs = {'filename': file_path}
        NeoBaseRecordingExtractor.__init__(self, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update({'file_path': file_path})


def read_axona(*args, **kwargs):
    recording = AxonaRecordingExtractor(*args, **kwargs)
    return recording


read_axona.__doc__ = AxonaRecordingExtractor.__doc__
