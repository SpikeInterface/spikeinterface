from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class SpikeGadgetsRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading *rec files from spikegadgets.

    Based on :py:class:`neo.rawio.SpikeGadgetsRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the one you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'SpikeGadgetsRawIO'

    def __init__(self, file_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update(dict(file_path=str(file_path), stream_id=stream_id))


read_spikegadgets = define_function_from_class(source_class=SpikeGadgetsRecordingExtractor, name="read_spikegadgets")
