from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import probeinterface as pi


class MEArecRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a MEArec simulated data.


    Parameters
    ----------
    file_path: str

    locs_2d: bool

    """
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'

    def __init__(self, file_path, locs_2d=True):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseRecordingExtractor.__init__(self, **neo_kwargs)

        probe = pi.read_mearec(file_path)
        self.set_probe(probe, in_place=True)
        self.annotate(is_filtered=True)

        self._kwargs = {'file_path': str(file_path), 'locs_2d': locs_2d}


class MEArecSortingExtractor(NeoBaseSortingExtractor):
    mode = 'file'
    NeoRawIOClass = 'MEArecRawIO'
    handle_spike_frame_directly = False

    def __init__(self, file_path, use_natural_unit_ids=True):
        neo_kwargs = {'filename': str(file_path)}
        NeoBaseSortingExtractor.__init__(self,
                                         sampling_frequency=None,  # auto guess is correct here
                                         use_natural_unit_ids=use_natural_unit_ids,
                                         **neo_kwargs)

        self._kwargs = {'file_path': str(file_path), 'use_natural_unit_ids': use_natural_unit_ids}


def read_mearec(file_path, locs_2d=True, use_natural_unit_ids=True):
    """

    Parameters
    ----------
    file_path: str or Path
        Path to MEArec h5 file
    locs_2d: bool
        If True (default), locations are loaded in 2d. If False, 3d locations are loaded
    use_natural_unit_ids: bool
        If True, natural unit strings are loaded (e.g. #0. #1). If False, unit ids are in64

    Returns
    -------
    recording: MEArecRecordingExtractor
        The recording extractor object
    sorting: MEArecSortingExtractor
        The sorting extractor object
    """
    recording = MEArecRecordingExtractor(file_path, locs_2d=locs_2d)
    sorting = MEArecSortingExtractor(file_path, use_natural_unit_ids=use_natural_unit_ids)
    return recording, sorting
