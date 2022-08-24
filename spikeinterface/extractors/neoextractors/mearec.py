import probeinterface as pi

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor
from .neo_utils import get_streams, get_num_blocks


class MEArecRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a MEArec simulated data.

    Based on :py:class:`neo.rawio.MEArecRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'

    def __init__(self, file_path, stream_id=None, stream_name=None, block_index=None,
                 all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, 
                                           stream_id=stream_id,
                                           stream_name=stream_name,
                                           block_index=block_index,
                                           all_annotations=all_annotations, **neo_kwargs)

        self.extra_requirements.append('mearec')

        probe = pi.read_mearec(file_path)
        self.set_probe(probe, in_place=True)
        self.annotate(is_filtered=True)

        if hasattr(self.neo_reader._recgen, "gain_to_uV"):
            self.set_channel_gains(self.neo_reader._recgen.gain_to_uV)

        self._kwargs.update({'file_path': str(file_path)})


class MEArecSortingExtractor(NeoBaseSortingExtractor):
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'
    handle_spike_frame_directly = False

    def __init__(self, file_path):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseSortingExtractor.__init__(self,
                                         sampling_frequency=None,  # auto guess is correct here
                                         use_natural_unit_ids=True,
                                         **neo_kwargs)

        self._kwargs = {'file_path': str(file_path)}


def read_mearec(file_path):
    """Read a MEArec file.

    Parameters
    ----------
    file_path: str or Path
        Path to MEArec h5 file

    Returns
    -------
    recording: MEArecRecordingExtractor
        The recording extractor object
    sorting: MEArecSortingExtractor
        The sorting extractor object
    """
    recording = MEArecRecordingExtractor(file_path)
    sorting = MEArecSortingExtractor(file_path)
    return recording, sorting


def get_mearec_streams(file_path):
    """Return available NEO streams

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.

    Returns
    -------
    list
        List of stream names
    list
        List of stream IDs
    """
    raw_class = MEArecRecordingExtractor.NeoRawIOClass
    neo_kwargs = {'filename': str(file_path)}
    return get_streams(raw_class, **neo_kwargs)


def get_mearec_num_blocks(file_path):
    """Return number of NEO blocks

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.

    Returns
    -------
    int
        Number of NEO blocks
    """
    raw_class = MEArecRecordingExtractor.NeoRawIOClass
    neo_kwargs = {'filename': str(file_path)}
    return get_num_blocks(raw_class, **neo_kwargs)
