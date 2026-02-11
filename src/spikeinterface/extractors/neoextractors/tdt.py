from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class TdtRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading TDT folder.

    Based on :py:class:`neo.rawio.TdTRawIO`

    Parameters
    ----------
    folder_path : str
        The folder path to the tdt folder.
    stream_id : str or None, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str or None, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.
    block_index : int, default: None
        If there are several blocks (experiments), specify the block index you want to load
    """

    NeoRawIOClass = "TdtRawIO"

    def __init__(
        self,
        folder_path,
        stream_id=None,
        stream_name=None,
        block_index=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            block_index=block_index,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )
        self._kwargs.update(dict(folder_path=str(Path(folder_path).absolute())))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        return neo_kwargs


read_tdt = define_function_from_class(source_class=TdtRecordingExtractor, name="read_tdt")
