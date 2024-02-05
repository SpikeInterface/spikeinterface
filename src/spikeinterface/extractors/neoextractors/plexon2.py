from __future__ import annotations

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor, NeoBaseEventExtractor


class Plexon2RecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading plexon pl2 files.

    Based on :py:class:`neo.rawio.Plexon2RawIO`

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
    NeoRawIOClass = "Plexon2RawIO"
    name = "plexon2"

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(
            self, stream_id=stream_id, stream_name=stream_name, all_annotations=all_annotations, **neo_kwargs
        )
        self._kwargs.update({"file_path": str(file_path)})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


class Plexon2SortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading plexon spiking data from .pl2 files.

    Based on :py:class:`neo.rawio.Plexon2RawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    sampling_frequency: float, default: None
        The sampling frequency of the sorting (required for multiple streams with different sampling frequencies).
    """

    mode = "file"
    NeoRawIOClass = "Plexon2RawIO"
    neo_returns_frames = True
    name = "plexon2"

    def __init__(self, file_path, sampling_frequency=None):
        from neo.rawio import Plexon2RawIO

        neo_kwargs = self.map_to_neo_kwargs(file_path)
        neo_reader = Plexon2RawIO(**neo_kwargs)
        neo_reader.parse_header()
        NeoBaseSortingExtractor.__init__(self, sampling_frequency=sampling_frequency, **neo_kwargs)
        self._kwargs.update({"file_path": str(file_path), "sampling_frequency": sampling_frequency})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


class Plexon2EventExtractor(NeoBaseEventExtractor):
    """
    Class for reading plexon spiking data from .pl2 files.

    Based on :py:class:`neo.rawio.Plexon2RawIO`

    Parameters
    ----------
    folder_path: str

    """

    mode = "file"
    NeoRawIOClass = "Plexon2RawIO"
    name = "plexon2"

    def __init__(self, folder_path, block_index=None):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseEventExtractor.__init__(self, block_index=block_index, **neo_kwargs)

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"filename": str(folder_path)}
        return neo_kwargs


read_plexon2 = define_function_from_class(source_class=Plexon2RecordingExtractor, name="read_plexon2")
read_plexon2_sorting = define_function_from_class(source_class=Plexon2SortingExtractor, name="read_plexon2_sorting")
read_plexon2_event = define_function_from_class(source_class=Plexon2EventExtractor, name="read_plexon2_event")
