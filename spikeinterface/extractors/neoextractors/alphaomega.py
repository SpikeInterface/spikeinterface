"""
"""
from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseEventExtractor


class AlphaOmegaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading from AlphaRS and AlphaLab SnR boards.

    Parameters
    ----------
    folder_path: str or Path-like
        The folder containing AlphaOmega recordings.
    stream_id: str, optional
        The stream to load. Can be `RAW` (default), `LFP`, `SPK`, `ACC`, `AI` or
        `UD`
    lsx_files: list of strings or None, optional
        A list of listings files that refers to mpx files to load
    """
    mode = "folder"
    NeoRawIOClass = "AlphaOmegaRawIO"

    def __init__(self, folder_path, stream_id="RAW", lsx_files=None):
        neo_kwargs = {
            "dirname": str(folder_path),
            "lsx_files": lsx_files,
        }
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)
        self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id)


class AlphaOmegaEventExtractor(NeoBaseEventExtractor):
    """
    Class for reading events from AlphaOmega MPX file format
    """
    mode = "folder"
    NeoRawIOClass = "AlphaOmegaRawIO"
    handle_event_frame_directly = True

    def __init__(self, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        NeoBaseEventExtractor.__init__(self, **neo_kwargs)


read_alphaomega = AlphaOmegaRecordingExtractor.define_reader_function(name="read_alphaomega")
read_alphaomega_event = AlphaOmegaEventExtractor.define_reader_function(name="read_alphaomega_event")
