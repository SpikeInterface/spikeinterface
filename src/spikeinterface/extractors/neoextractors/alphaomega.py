from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseEventExtractor


class AlphaOmegaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading from AlphaRS and AlphaLab SnR boards.

    Based on :py:class:`neo.rawio.AlphaOmegaRawIO`

    Parameters
    ----------
    folder_path : str or Path-like
        The folder path to the AlphaOmega recordings.
    lsx_files : list of strings or None, default: None
        A list of files that refers to mpx files to load.
    stream_id : {"RAW", "LFP", "SPK", "ACC", "AI", "UD"}, default: "RAW"
        If there are several streams, specify the stream id you want to load.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.

    Examples
    --------
    >>> from spikeinterface.extractors import read_alphaomega
    >>> recording = read_alphaomega(folder_path="alphaomega_folder")

    """

    NeoRawIOClass = "AlphaOmegaRawIO"

    def __init__(
        self,
        folder_path,
        lsx_files=None,
        stream_id="RAW",
        stream_name=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, lsx_files)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )
        self._kwargs.update(dict(folder_path=str(Path(folder_path).absolute()), lsx_files=lsx_files))

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

    Parameters
    ----------
    folder_path : str or Path-like
        The folder path to the AlphaOmega events.
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
