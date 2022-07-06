from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import probeinterface as pi


class MEArecRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a MEArec simulated data.


    Parameters
    ----------
    file_path: str

    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
    """
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'

    def __init__(self, file_path, all_annotations=False):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, all_annotations=all_annotations, **neo_kwargs)

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
    """
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
