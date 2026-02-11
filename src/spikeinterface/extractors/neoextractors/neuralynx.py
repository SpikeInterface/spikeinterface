from __future__ import annotations

from typing import Optional
from pathlib import Path

from importlib.metadata import version

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading neuralynx folder

    Based on :py:class:`neo.rawio.NeuralynxRawIO`

    Parameters
    ----------
    folder_path : str
        The file path to load the recordings from.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    exclude_filename : list[str], default: None
        List of filename to exclude from the loading.
        For example, use `exclude_filename=["events.nev"]` to skip loading the event file.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.
    strict_gap_mode : bool, default: False
        See neo documentation.
        Detect gaps using strict mode or not.
        * strict_gap_mode = True then a gap is consider when timstamp difference between
        two consecutive data packets is more than one sample interval.
        * strict_gap_mode = False then a gap has an increased tolerance. Some new systems
        with different clocks need this option otherwise, too many gaps are detected

        Note that here the default is False contrary to neo.
    """

    NeoRawIOClass = "NeuralynxRawIO"

    def __init__(
        self,
        folder_path: str | Path,
        stream_id=None,
        stream_name=None,
        all_annotations=False,
        exclude_filename=None,
        strict_gap_mode=False,
        use_names_as_ids: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, exclude_filename, strict_gap_mode)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )
        self._kwargs.update(
            dict(folder_path=str(Path(folder_path).absolute()), exclude_filename=exclude_filename),
            strict_gap_mode=strict_gap_mode,
        )

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, exclude_filename, strict_gap_mode):
        neo_kwargs = {"dirname": str(folder_path), "exclude_filename": exclude_filename}
        if version("neo") >= "0.13.1":
            neo_kwargs["strict_gap_mode"] = strict_gap_mode

        return neo_kwargs


class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading spike data from a folder with neuralynx spiking data (i.e .nse and .ntt formats).

    Based on :py:class:`neo.rawio.NeuralynxRawIO`

    Parameters
    ----------
    folder_path : str
        The file path to load the recordings from.
    sampling_frequency : float
        The sampling frequency for the spiking channels. When the signal data is available (.ncs) those files will be
        used to extract the frequency. Otherwise, the sampling frequency needs to be specified for this extractor.
    stream_id : str, default: None
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    stream_name : str, default: None
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    """

    NeoRawIOClass = "NeuralynxRawIO"
    neo_returns_frames = True
    need_t_start_from_signal_stream = True

    def __init__(
        self,
        folder_path: str,
        sampling_frequency: Optional[float] = None,
        stream_id: Optional[str] = None,
        stream_name: Optional[str] = None,
    ):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseSortingExtractor.__init__(
            self,
            sampling_frequency=sampling_frequency,
            stream_id=stream_id,
            stream_name=stream_name,
            **neo_kwargs,
        )

        self._kwargs = {
            "folder_path": str(Path(folder_path).absolute()),
            "sampling_frequency": sampling_frequency,
            "stream_id": stream_id,
            "stream_name": stream_name,
        }

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        return neo_kwargs


read_neuralynx = define_function_from_class(source_class=NeuralynxRecordingExtractor, name="read_neuralynx")
read_neuralynx_sorting = define_function_from_class(
    source_class=NeuralynxSortingExtractor, name="read_neuralynx_sorting"
)
