from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class NixRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Nix file
    
    Based on neo.rawio.NIXRawIO
    
    Parameters
    ----------
    file_path: str
    
    stream_id: str or None
    """
    mode = 'file'
    NeoRawIOClass = 'NIXRawIO'

    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id)


def read_nix(*args, **kwargs):
    recording = NixRecordingExtractor(*args, **kwargs)
    return recording


read_nix.__doc__ = NixRecordingExtractor.__doc__
