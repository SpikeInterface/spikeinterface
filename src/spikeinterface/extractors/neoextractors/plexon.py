from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class PlexonRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading plexon plx files.

    Based on :py:class:`neo.rawio.PlexonRawIO`

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
    NeoRawIOClass = "PlexonRawIO"
    name = "plexon"

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(
            self, stream_id=stream_id, stream_name=stream_name, all_annotations=all_annotations, **neo_kwargs
        )
        self._kwargs.update({"file_path": str(Path(file_path).resolve())})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


class PlexonSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading plexon spiking data (.plx files).

    Based on :py:class:`neo.rawio.PlexonRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    """

    mode = "file"
    NeoRawIOClass = "PlexonRawIO"
    name = "plexon"
    neo_returns_frames = True

    def __init__(self, file_path):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        self.neo_reader = NeoBaseSortingExtractor.get_neo_io_reader(self.NeoRawIOClass, **neo_kwargs)
        sampling_frequency = self.neo_reader._global_ssampling_rate
        NeoBaseSortingExtractor.__init__(self, sampling_frequency=sampling_frequency, **neo_kwargs)
        self._kwargs = {"file_path": str(Path(file_path).resolve())}

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_plexon = define_function_from_class(source_class=PlexonRecordingExtractor, name="read_plexon")
read_plexon_sorting = define_function_from_class(source_class=PlexonSortingExtractor, name="read_plexon_sorting")
