from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Axona RAW format

    Based on :py:class:`neo.rawio.AxonaRawIO`

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


read_axona = define_function_from_class(source_class=AxonaRecordingExtractor, name="read_axona")
