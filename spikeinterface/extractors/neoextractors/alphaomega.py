from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseEventExtractor


class AlphaOmegaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading from AlphaRS and AlphaLab SnR boards.

    Based on :py:class:`neo.rawio.AlphaOmegaRawIO`

    Parameters
    ----------
    folder_path: str or Path-like
        The folder path to the AlphaOmega recordings.
    lsx_files: list of strings or None, optional
        A list of listings files that refers to mpx files to load.
    stream_id: {'RAW', 'LFP', 'SPK', 'ACC', 'AI', 'UD'}, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = "folder"
    NeoRawIOClass = "AlphaOmegaRawIO"
    name = "alphaomega"

    def __init__(self, folder_path, lsx_files=None, stream_id="RAW", 
                 stream_name=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, lsx_files)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations,
                                           **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path), lsx_files=lsx_files))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, lsx_files=None):
        neo_kwargs = {
            "dirname": str(folder_path),
            "lsx_files": lsx_files,
        }
        return neo_kwargs


class AlphaOmegaEventExtractor(NeoBaseEventExtractor):
    """
    Class for reading events from AlphaOmega MPX file format
    """
    mode = "folder"
    NeoRawIOClass = "AlphaOmegaRawIO"
    handle_event_frame_directly = True

    def __init__(self, folder_path):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseEventExtractor.__init__(self, **neo_kwargs)

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        return neo_kwargs


read_alphaomega = define_function_from_class(source_class=AlphaOmegaRecordingExtractor, name="read_alphaomega")
read_alphaomega_event = define_function_from_class(source_class=AlphaOmegaEventExtractor, name="read_alphaomega_event")
