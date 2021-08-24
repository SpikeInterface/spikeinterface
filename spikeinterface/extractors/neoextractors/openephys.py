"""
There are 2 openephys reader:
  * OpenEphysLegacyRecordingExtractor: old one aka "open ephys format"
  * OpenEphysBinaryRecordingExtractor: new one aka "binary format"

https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/index.html
"""
from pathlib import Path

import numpy as np

from .neobaseextractor import (NeoBaseRecordingExtractor,
                               NeoBaseSortingExtractor,
                               NeoBaseEventExtractor)


class OpenEphysLegacyRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a OpenEphy board.
    
    This open the openephys "legacy" format: one file per channel.
    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Open-Ephys-format.html
    
    Based on neo.rawio.OpenEphysRawIO
    
    Parameters
    ----------
    folder_path: str
        
    stream_id: str or None
        stream for instance : 'imec0.ap' 'nidq' or 'imec0.lf'
    """
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysRawIO'

    def __init__(self, folder_path, stream_id=None):
        neo_kwargs = {'dirname': folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id)


class OpenEphysBinaryRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading traces from a OpenEphy board.
    
    This open the openephys "new" "binary" format: one file per continuous stream.
    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html
    
    Based on neo.rawio.OpenEphysBinaryRawIO
    
    Parameters
    ----------
    folder_path: str
        
    stream_id: str or None
        stream for instance : 'imec0.ap' 'nidq' or 'imec0.lf'
    """
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysBinaryRawIO'

    def __init__(self, folder_path, stream_id=None):
        neo_kwargs = {'dirname': folder_path}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

        self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id)


class OpenEphysBinaryEventExtractor(NeoBaseEventExtractor):
    """
    Class for reading events from a OpenEphy board.
    
    This open the openephys "new" "binary" format: one file per continuous stream.
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
    if np.any([f.startswith('Continuous') for f in files]):
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
