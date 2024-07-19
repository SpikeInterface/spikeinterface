from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class CedRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading smr/smrw CED file.

    Based on :py:class:`neo.rawio.CedRawIO` / sonpy

    Alternative to read_spike2 which does not handle smrx

    Parameters
    ----------
    file_path : str
        The file path to the smr or smrx file.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.
    """

    NeoRawIOClass = "CedRawIO"

    def __init__(
        self, file_path, stream_id=None, stream_name=None, all_annotations: bool = False, use_names_as_ids: bool = False
    ):
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
        self.extra_requirements.append("neo[ced]")

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_ced = define_function_from_class(source_class=CedRecordingExtractor, name="read_ced")
