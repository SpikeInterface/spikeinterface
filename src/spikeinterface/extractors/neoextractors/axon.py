from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class AxonRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading Axon Binary Format (ABF) files.

    Based on :py:class:`neo.rawio.AxonRawIO`

    Supports both ABF1 (pClamp ≤9) and ABF2 (pClamp ≥10) formats.
    Can read data from pCLAMP and AxoScope software.

    Parameters
    ----------
    file_path : str or Path
        The ABF file path to load the recordings from.
    stream_id : str or None, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str or None, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor.

    Examples
    --------
    >>> from spikeinterface.extractors import read_axon
    >>> recording = read_axon(file_path='path/to/file.abf')
    """

    NeoRawIOClass = "AxonRawIO"

    def __init__(
        self,
        file_path,
        stream_id=None,
        stream_name=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
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

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_axon = define_function_from_class(source_class=AxonRecordingExtractor, name="read_axon")
