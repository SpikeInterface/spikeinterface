from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class Spike2RecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading spike2 smr files.
    smrx are not supported with this, prefer CedRecordingExtractor instead.
    
    Based on neo.rawio.Spike2RawIO
    
    Parameters
    ----------
    file_path: str
        The xml  file.
    stream_id: str or None
    """
    mode = 'file'
    NeoRawIOClass = 'Spike2RawIO'

    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename': file_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self._kwargs = {'file_path': str(file_path), 'stream_id': stream_id}


def read_spike2(*args, **kwargs):
    recording = Spike2RecordingExtractor(*args, **kwargs)
    return recording


read_spike2.__doc__ = Spike2RecordingExtractor.__doc__
