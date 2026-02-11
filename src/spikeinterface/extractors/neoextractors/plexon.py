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
    file_path : str | Path
        The file path to load the recordings from.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: True
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.

        Example for wideband signals:
            names: ["WB01", "WB02", "WB03", "WB04"]
            ids: ["0" , "1", "2", "3"]

    Examples
    --------
    >>> from spikeinterface.extractors import read_plexon
    >>> recording = read_plexon(file_path=r'my_data.plx')
    """

    NeoRawIOClass = "PlexonRawIO"

    def __init__(
        self,
        file_path: str | Path,
        stream_id=None,
        stream_name=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = True,
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
    file_path : str
        The file path to load the recordings from.
    """

    NeoRawIOClass = "PlexonRawIO"
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
