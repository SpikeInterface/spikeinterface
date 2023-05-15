from pathlib import Path
from packaging import version
from typing import Optional

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
    mode = 'file'
    NeoRawIOClass = 'BlackrockRawIO'
    name = "blackrock"

    def __init__(self, file_path, stream_id=None, stream_name=None, block_index=None,
                 all_annotations=False, use_names_as_ids=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        if version.parse(neo.__version__) > version.parse('0.12.0'):
            # do not load spike because this is slow but not released yet
            neo_kwargs['load_nev'] = False
        # trick to avoid to select automatically the correct stream_id
        suffix = Path(file_path).suffix
        if '.ns' in suffix:
            neo_kwargs['nsx_to_load'] = int(suffix[-1])
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations,
                                           use_names_as_ids=use_names_as_ids,
                                           **neo_kwargs)
        self._kwargs.update({'file_path': str(file_path)})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {'filename': str(file_path)}
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
    """
    
    mode = 'file'
    NeoRawIOClass = 'BlackrockRawIO'
    neo_returns_timestamps = True
    name = "blackrock"

    def __init__(self, file_path, stream_index: Optional[int] = None):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        
        if stream_index is None:
            stream_names, stream_ids = self.get_streams(file_path)
            
            error_msg = (
                "Black rock requires analog signal streams nsx5 or nsx6 to be present in for inferring spike times"
            )
            
            # Return the index of the first stream that is named nsx5 or nsx6 in that order
            stream_index = next((index for index, name in enumerate(stream_names) if name == "nsx5"), None)
            if stream_index is None:
                stream_index = next((index for index, name in enumerate(stream_names) if name == "nsx6"), None)
            assert stream_index is not None, error_msg            
            stream_index = 0
        
        NeoBaseSortingExtractor.__init__(self, **neo_kwargs, stream_index=stream_index)
        self._kwargs.update({'file_path': str(file_path)})

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {'filename': str(file_path)}
        return neo_kwargs

read_blackrock = define_function_from_class(source_class=BlackrockRecordingExtractor, name="read_blackrock")
read_blackrock_sorting = define_function_from_class(source_class=BlackrockSortingExtractor,
                                                    name="read_blackrock_sorting")
