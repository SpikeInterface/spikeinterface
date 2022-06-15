from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading plexon plx files.
    
    Based on :py:class:`neo.rawio.PlexonRawIO`
    
    Parameters
    ----------
    file_path: str
        The xml  file.
    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'PlexonRawIO'

    def __init__(self, file_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'filename': file_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update({'file_path': str(file_path)})


read_plexon = define_function_from_class(source_class=PlexonRecordingExtractor, name="read_plexon")
