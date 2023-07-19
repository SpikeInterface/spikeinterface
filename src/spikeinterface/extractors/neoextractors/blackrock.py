from pathlib import Path
from packaging import version
from typing import Optional

import neo

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class BlackrockRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading BlackRock data.

    Based on :py:class:`neo.rawio.BlackrockRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, default: False
        Load exhaustively all annotations from neo.
    """

    mode = "file"
    NeoRawIOClass = "BlackrockRawIO"
    name = "blackrock"

    def __init__(
        self,
        file_path,
        stream_id=None,
        stream_name=None,
        block_index=None,
        all_annotations=False,
        use_names_as_ids=False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        if version.parse(neo.__version__) > version.parse("0.12.0"):
            # do not load spike because this is slow but not released yet
            neo_kwargs["load_nev"] = False
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
    file_path: str
        The file path to load the recordings from.
    sampling_frequency: float, None by default.
        The sampling frequency for the sorting extractor. When the signal data is available (.ncs) those files will be
        used to extract the frequency automatically. Otherwise, the sampling frequency needs to be specified for
        this extractor to be initialized.
    stream_id: str, optional
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    stream_name: str, optional
        Used to extract information about the sampling frequency and t_start from the analog signal if provided.
    """

    mode = "file"
    NeoRawIOClass = "BlackrockRawIO"
    neo_returns_frames = False
    name = "blackrock"

    def __init__(
        self,
        file_path,
        sampling_frequency: Optional[float] = None,
        stream_id: Optional[str] = None,
        stream_name: Optional[str] = None,
    ):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
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


read_blackrock = define_function_from_class(source_class=BlackrockRecordingExtractor, name="read_blackrock")
read_blackrock_sorting = define_function_from_class(
    source_class=BlackrockSortingExtractor, name="read_blackrock_sorting"
)
