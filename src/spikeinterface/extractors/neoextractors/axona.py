from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class AxonaRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Axona RAW format.

    Based on :py:class:`neo.rawio.AxonaRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    all_annotations: bool, default: False
        Load exhaustively all annotations from neo.
    """

    mode = "folder"
    NeoRawIOClass = "AxonaRawIO"
    name = "axona"

    def __init__(self, file_path, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(self, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update({"file_path": str(Path(file_path).absolute())})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_axona = define_function_from_class(source_class=AxonaRecordingExtractor, name="read_axona")
