from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class IntanRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a intan board support rhd and rhs format.
    
    Based on :py:class:`neo.rawio.IntanRawIO`
    
    Parameters
    ----------
    file_path: str
        
    stream_id: str or None
        
    """
    mode = 'file'
    NeoRawIOClass = 'IntanRawIO'

    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id)


read_intan = define_function_from_class(source_class=IntanRecordingExtractor, name="read_intan")