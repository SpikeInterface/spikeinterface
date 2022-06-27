"""

There are two extractors for data saved by the Open Ephys GUI

  * OpenEphysLegacyRecordingExtractor: reads the original "Open Ephys" data format
  * OpenEphysBinaryRecordingExtractor: reads the new default "Binary" format

See https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/index.html 
for more info.

"""

from pathlib import Path

import numpy as np

import probeinterface as pi

from .neobaseextractor import (NeoBaseRecordingExtractor,
                               NeoBaseSortingExtractor,
                               NeoBaseEventExtractor)


class OpenEphysLegacyRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data saved by the Open Ephys GUI.

    This extractor works with the Open Ephys "legacy" format, which saves data using
    one file per continuous channel (.continuous files).

    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Open-Ephys-format.html

    Based on :py:class:`neo.rawio.OpenEphysRawIO`

    Parameters
    ----------
    folder_path: str

    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.

    """
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysRawIO'

    def __init__(self, folder_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'dirname': folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path), stream_id=stream_id))


class OpenEphysBinaryRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data saved by the Open Ephys GUI.

    This extractor works with the  Open Ephys "binary" format, which saves data using
    one file per continuous stream (.dat files).

    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html

    Based on neo.rawio.OpenEphysBinaryRawIO

    Parameters
    ----------
    folder_path: str

    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
        
    """
    
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysBinaryRawIO'

    def __init__(self, folder_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'dirname': folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)

        probe = pi.read_openephys(folder_path, raise_error=False)
        if probe is not None:
            self.set_probe(probe, in_place=True)
            
        self._kwargs .update(dict(folder_path=str(folder_path)))


class OpenEphysBinaryEventExtractor(NeoBaseEventExtractor):
    """
    Class for reading events saved by the Open Ephys GUI

    This extractor works with the  Open Ephys "binary" format, which saves data using
    one file per continuous stream.

    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html

    Based on neo.rawio.OpenEphysBinaryRawIO

    Parameters
    ----------
    folder_path: str

    """
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysBinaryRawIO'

    def __init__(self, folder_path):
        neo_kwargs = {'dirname': str(folder_path)}
        NeoBaseEventExtractor.__init__(self, **neo_kwargs)


def read_openephys(folder_path, **kwargs):
    """
    Read 'legacy' or 'binary' Open Ephys formats

    Parameters
    ----------
    folder_path: str or Path
        Path to openephys folder

    Returns
    -------
    recording: OpenEphysLegacyRecordingExtractor or OpenEphysBinaryExtractor
    """
    # auto guess format
    files = [str(f) for f in Path(folder_path).iterdir()]
    if np.any([f.endswith('continuous') for f in files]):
        #  format = 'legacy'
        recording = OpenEphysLegacyRecordingExtractor(folder_path, **kwargs)
    else:
        # format = 'binary'
        recording = OpenEphysBinaryRecordingExtractor(folder_path, **kwargs)
    return recording


def read_openephys_event(folder_path, **kwargs):
    """
    Read Open Ephys events from 'binary' format.

    Parameters
    ----------
    folder_path: str or Path
        Path to openephys folder

    Returns
    -------
    event: OpenEphysBinaryEventExtractor
    """
    # auto guess format
    files = [str(f) for f in Path(folder_path).iterdir()]
    if np.any([f.startswith('Continuous') for f in files]):
        raise Exception("Events can be read only from 'binary' format")
    else:
        # format = 'binary'
        event = OpenEphysBinaryEventExtractor(folder_path, **kwargs)
    return event
