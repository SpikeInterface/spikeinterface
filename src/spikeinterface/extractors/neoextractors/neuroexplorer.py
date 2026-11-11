from pathlib import Path

from spikeinterface.core import BaseSorting
from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import (
    NeoBaseRecordingExtractor,
    NeoBaseSortingExtractor,
    NeoSortingSegment,
    _NeoBaseExtractor,
)


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


class NeuroExplorerSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading the sorted units of NEX (NeuroExplorer data format) files.

    Based on :py:class:`neo.rawio.NeuroExplorerRawIO`

    Units are the neuron variables of the file and the unit ids are their names. Waveform variables
    (for example the ``_wf`` and ``_template`` companions of each unit in Plexon exports) repeat
    the spike times of a neuron variable or hold a mean waveform, so they are not units. They
    remain available through ``neo_reader``.

    Parameters
    ----------
    file_path : str | Path
        The file path to load the sorting from.
    """

    NeoRawIOClass = "NeuroExplorerRawIO"
    neo_returns_frames = True

    def __init__(self, file_path):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        _NeoBaseExtractor.__init__(self, block_index=None, **neo_kwargs)

        # Only neuron variables are units: they are the spike channels without waveforms
        spike_channels = self.neo_reader.header["spike_channels"]
        unit_ids = spike_channels["name"][spike_channels["wf_sampling_rate"] == 0]
        # Spike timestamps are ticks of the file's global clock
        sampling_frequency = self.neo_reader.global_header["freq"]
        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        sorting_segment = NeuroExplorerSortingSegment(
            neo_reader=self.neo_reader,
            block_index=self.block_index,
            segment_index=0,
            t_start=None,
            sampling_frequency=sampling_frequency,
            neo_returns_frames=self.neo_returns_frames,
        )
        self.add_sorting_segment(sorting_segment)
        self._kwargs = {"file_path": str(Path(file_path).absolute())}

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


class NeuroExplorerSortingSegment(NeoSortingSegment):
    def map_from_unit_id_to_spike_channel_index(self, unit_id):
        # Units are a subset of the spike channels, so the position of a unit id is not its spike channel index
        spike_channel_names = list(self.neo_reader.header["spike_channels"]["name"])
        return spike_channel_names.index(unit_id)


read_neuroexplorer_sorting = define_function_from_class(
    source_class=NeuroExplorerSortingExtractor, name="read_neuroexplorer_sorting"
)
