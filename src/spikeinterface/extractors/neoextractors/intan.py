from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class IntanRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a intan board. Supports rhd and rhs format.

    Based on :py:class:`neo.rawio.IntanRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name: str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, default: False
        Load exhaustively all annotations from neo.
    """

    mode = "file"
    NeoRawIOClass = "IntanRawIO"
    name = "intan"

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False, use_names_as_ids=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )

        self._kwargs.update(dict(file_path=str(Path(file_path).absolute())))

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_intan = define_function_from_class(source_class=IntanRecordingExtractor, name="read_intan")
