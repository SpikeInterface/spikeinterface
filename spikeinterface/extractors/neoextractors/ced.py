from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class CedRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading smr/smrw CED file.

    Based on neo.rawio.CedRawIO / sonpy
    
    Alternative to read_spike2 which do not handle smrx
    
    Parameters
    ----------
    file_path: str
        The smr or smrx  file.
    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'CedRawIO'

    def __init__(self, file_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update(dict(file_path=str(file_path)))


def read_ced(*args, **kwargs):
    recording = CedRecordingExtractor(*args, **kwargs)
    return recording


read_ced.__doc__ = CedRecordingExtractor.__doc__
