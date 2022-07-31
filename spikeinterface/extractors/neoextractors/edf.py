from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor

class EDFRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading EDF (European data format) folder.

    Based on :py:class:`neo.rawio.EDFRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the one you want to load.
        For this neo reader stream are defined by their sampling frequency.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'EDFRawIO'

    def __init__(self, file_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update({'file_path': str(file_path)})
        self.extra_requirements.append('pyedflib')


read_edf = define_function_from_class(source_class=EDFRecordingExtractor, name="read_edf")