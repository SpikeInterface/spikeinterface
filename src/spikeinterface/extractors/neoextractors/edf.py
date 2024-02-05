from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class EDFRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading EDF (European data format) folder.

    Based on :py:class:`neo.rawio.EDFRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, default: None
        If there are several streams, specify the stream id you want to load.
        For this neo reader streams are defined by their sampling frequency.
    stream_name: str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, default: False
        Load exhaustively all annotations from neo.
    """

    mode = "file"
    NeoRawIOClass = "EDFRawIO"
    name = "edf"

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = {"filename": str(file_path)}
        NeoBaseRecordingExtractor.__init__(
            self, stream_id=stream_id, stream_name=stream_name, all_annotations=all_annotations, **neo_kwargs
        )
        self._kwargs.update({"file_path": str(Path(file_path).absolute())})
        self.extra_requirements.append("neo[edf]")

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_edf = define_function_from_class(source_class=EDFRecordingExtractor, name="read_edf")
