from __future__ import annotations

from typing import Optional
from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class NeuralynxRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading neuralynx folder

    Based on :py:class:`neo.rawio.NeuralynxRawIO`

    Parameters
    ----------
    folder_path: str
        The file path to load the recordings from.
    stream_id: str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name: str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, default: False
        Load exhaustively all annotations from neo.
    exlude_filename: list[str], default: None
        List of filename to exclude from the loading.
        For example, use `exclude_filename=["events.nev"]` to skip loading the event file.
    """

    mode = "folder"
    NeoRawIOClass = "NeuralynxRawIO"
    name = "neuralynx"

    def __init__(self, folder_path, stream_id=None, stream_name=None, all_annotations=False, exclude_filename=None):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, exclude_filename)
        NeoBaseRecordingExtractor.__init__(
            self, stream_id=stream_id, stream_name=stream_name, all_annotations=all_annotations, **neo_kwargs
        )
        self._kwargs.update(dict(folder_path=str(Path(folder_path).absolute()), exclude_filename=exclude_filename))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, exclude_filename):
        neo_kwargs = {"dirname": str(folder_path), "exclude_filename": exclude_filename}
        return neo_kwargs


class NeuralynxSortingExtractor(NeoBaseSortingExtractor):
    """
    Class for reading spike data from a folder with neuralynx spiking data (i.e .nse and .ntt formats).

    Based on :py:class:`neo.rawio.NeuralynxRawIO`

    Parameters
    ----------
    folder_path: str
        The file path to load the recordings from.
    sampling_frequency: float
        The sampling frequency for the spiking channels. When the signal data is available (.ncs) those files will be
        used to extract the frequency. Otherwise, the sampling frequency needs to be specified for this extractor.
    stream_id: str, default: None
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    stream_name: str, default: None
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    """

    mode = "folder"
    NeoRawIOClass = "NeuralynxRawIO"
    neo_returns_frames = True
    need_t_start_from_signal_stream = True
    name = "neuralynx"

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
