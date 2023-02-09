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
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'PlexonRawIO'
    name = "plexon"

    def __init__(self, file_path, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations, 
                                           **neo_kwargs)
        self._kwargs.update({'file_path': str(file_path)})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {'filename': str(file_path)}
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
    handle_spike_frame_directly = False
    name = "plexon"

    def __init__(self, file_path):
        from neo.rawio import PlexonRawIO
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        neo_reader = PlexonRawIO(**neo_kwargs)
        neo_reader.parse_header()
        sampling_frequency = neo_reader._global_ssampling_rate
        NeoBaseSortingExtractor.__init__(self, sampling_frequency=sampling_frequency,
                                         **neo_kwargs)
        self._kwargs.update({"file_path": str(file_path)})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_plexon = define_function_from_class(source_class=PlexonRecordingExtractor, name="read_plexon")
read_plexon_sorting = define_function_from_class(source_class=PlexonSortingExtractor, name="read_plexon_sorting")
