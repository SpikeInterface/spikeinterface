from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class IntanRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a intan board support rhd and rhs format.
    
    Based on neo.rawio.IntanRawIO
    
    Parameters
    ----------
    file_path: str
        
    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'IntanRawIO'

    def __init__(self, file_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)

        self._kwargs.update(dict(file_path=str(file_path)))


def read_intan(*args, **kwargs):
    recording = IntanRecordingExtractor(*args, **kwargs)
    return recording


read_intan.__doc__ = IntanRecordingExtractor.__doc__
