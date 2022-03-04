from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Axona RAW format

    Based on neo.rawio.AxonaRawIO

    Parameters
    ----------
    folder_path: str
        The tdt folder.
    stream_id: str or None
    """
    mode = 'folder'
    NeoRawIOClass = 'AxonaRawIO'

    def __init__(self, file_path):
        neo_kwargs = {'filename': file_path}
        NeoBaseRecordingExtractor.__init__(self, **neo_kwargs)

        self._kwargs = dict(file_path=str(file_path))


def read_axona(*args, **kwargs):
    recording = AxonaRecordingExtractor(*args, **kwargs)
    return recording


read_axona.__doc__ = AxonaRecordingExtractor.__doc__
