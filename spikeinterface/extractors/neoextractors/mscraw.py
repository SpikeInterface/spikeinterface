from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class MCSRawRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from "Raw" Multi Channel System (MCS) format.
    This format is NOT the native MCS format (*.mcd).
    This format is a raw format with an internal binary header exported by the
    "MC_DataTool binary conversion" with the option header selected.
    
    Based on neo.rawio.NeuralynxRawIO
    
    Parameters
    ----------
    file_path: str
        The xml  file.
    stream_id: str or None
    """
    mode = 'file'
    NeoRawIOClass = 'RawMCSRawIO'

    def __init__(self, file_path, stream_id=None):
        neo_kwargs = {'filename': file_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self._kwargs = dict(file_path=str(file_path), stream_id=stream_id)


read_mcsraw = define_function_from_class(source_class=MCSRawRecordingExtractor, name="read_maxwell_event")
