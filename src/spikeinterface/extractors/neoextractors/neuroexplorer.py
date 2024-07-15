from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class NeuroExplorerRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading NEX (NeuroExplorer data format) files.

    Based on :py:class:`neo.rawio.NeuroExplorerRawIO`

    Importantly, at the moment, this recorder only extracts one channel of the recording.
    This is because the NeuroExplorerRawIO class does not support multi-channel recordings
    as in the NeuroExplorer format they might have different sampling rates.

    Consider extracting all the channels and then concatenating them with the aggregate_channels function.

    >>> from spikeinterface.extractors.neoextractors.neuroexplorer import NeuroExplorerRecordingExtractor
    >>> from spikeinterface.core import aggregate_channels
    >>>
    >>> file_path="/the/path/to/your/nex/file.nex"
    >>>
    >>> streams = NeuroExplorerRecordingExtractor.get_streams(file_path=file_path)
    >>> stream_names = streams[0]
    >>>
    >>> your_signal_stream_names = "Here goes the logic to filter from stream names the ones that you know have the same sampling rate and you want to aggregate"
    >>>
    >>> recording_list = [NeuroExplorerRecordingExtractor(file_path=file_path, stream_name=stream_name) for stream_name in your_signal_stream_names]
    >>> recording = aggregate_channels(recording_list)



    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
        For this neo reader streams are defined by their sampling frequency.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.
    """

    NeoRawIOClass = "NeuroExplorerRawIO"

    def __init__(
        self, file_path, stream_id=None, stream_name=None, all_annotations: bool = False, use_names_as_ids: bool = False
    ):
        neo_kwargs = {"filename": str(file_path)}
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )
        self._kwargs.update({"file_path": str(Path(file_path).absolute())})
        self.extra_requirements.append("neo[edf]")

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_neuroexplorer = define_function_from_class(source_class=NeuroExplorerRecordingExtractor, name="read_neuroexplorer")
