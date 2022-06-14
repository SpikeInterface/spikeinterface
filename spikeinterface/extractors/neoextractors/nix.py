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
    """
    mode = 'file'
    NeoRawIOClass = 'NIXRawIO'

    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self.extra_requirements.append('nixio')

        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id)


read_nix = define_function_from_class(source_class=NixRecordingExtractor, name="read_nix")
