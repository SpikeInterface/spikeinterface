from __future__ import annotations

from pathlib import Path
from packaging import version
from typing import Optional, Dict
import numpy as np

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class BlackrockRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading BlackRock data.

    Based on :py:class:`neo.rawio.BlackrockRawIO`

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.
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

    NeoRawIOClass = "BlackrockRawIO"

    def __init__(
        self,
        file_path,
        stream_id=None,
        stream_name=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        neo_kwargs["load_nev"] = False  # Avoid loading spikes release in neo 0.12.0

        # trick to avoid to select automatically the correct stream_id
        suffix = Path(file_path).suffix
        if ".ns" in suffix:
            neo_kwargs["nsx_to_load"] = int(suffix[-1])
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )
        self._kwargs.update({"file_path": str(Path(file_path).absolute())})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


class BlackrockSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading BlackRock spiking data.

    Based on :py:class:`neo.rawio.BlackrockRawIO`


    Parameters
    ----------
    file_path : str
        The file path to load the recordings from
    sampling_frequency : float, default: None
        The sampling frequency for the sorting extractor. When the signal data is available (.ncs) those files will be
        used to extract the frequency automatically. Otherwise, the sampling frequency needs to be specified for
        this extractor to be initialized
    stream_id : str, default: None
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    stream_name : str, default: None
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    """

    NeoRawIOClass = "BlackrockRawIO"
    neo_returns_frames = False

    def __init__(
        self,
        file_path,
        sampling_frequency: Optional[float] = None,
        stream_id: Optional[str] = None,
        stream_name: Optional[str] = None,
    ):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        neo_kwargs["nsx_to_load"] = []  # Tell neo to not load the nsx files
        NeoBaseSortingExtractor.__init__(
            self,
            sampling_frequency=sampling_frequency,
            stream_id=stream_id,
            stream_name=stream_name,
            **neo_kwargs,
        )

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "sampling_frequency": sampling_frequency,
            "stream_id": stream_id,
            "stream_name": stream_name,
        }

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs

    def build_stream_id_to_sampling_frequency_dict(self) -> Dict[str, float]:
        """
        Build a mapping from stream_id to sampling frequencies.

        This function creates a dictionary mapping each stream_id to its corresponding sampling
        frequency, as extracted from the signal channels in the Neo header.

        Returns
        -------
        dict of {str: float}
            Dictionary mapping stream_ids to their corresponding sampling frequencies.

        Raises
        ------
        AssertionError
            If there are no signal streams available from which to extract the sampling frequencies.
        """
        neo_header = self.neo_reader.header
        if "spike_channels" in neo_header and neo_header["spike_channels"].size > 0:
            channels = neo_header["spike_channels"]
            channel_sampling_frequencies = channels["wf_sampling_rate"]
            stream_to_sampling_frequencies = {0: float(np.unique(channel_sampling_frequencies)[0])}
        else:
            raise AssertionError("No signal or spike streams to infer the sampling frequency. Set it manually")

        return stream_to_sampling_frequencies


read_blackrock = define_function_from_class(source_class=BlackrockRecordingExtractor, name="read_blackrock")
read_blackrock_sorting = define_function_from_class(
    source_class=BlackrockSortingExtractor, name="read_blackrock_sorting"
)
