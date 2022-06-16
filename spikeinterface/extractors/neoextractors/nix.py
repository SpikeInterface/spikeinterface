from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

try:
    import nixio
    HAVE_NIX = True
except ModuleNotFoundError:
    HAVE_NIX = False

class NixRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Nix file
    
    Based on neo.rawio.NIXRawIO
    
    Parameters
    ----------
    file_path: str
    
    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'NIXRawIO'

    def __init__(self, file_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update(dict(file_path=str(file_path), stream_id=stream_id))
        self.extra_requirements.append('nixio')



read_nix = define_function_from_class(source_class=NixRecordingExtractor, name="read_nix")
