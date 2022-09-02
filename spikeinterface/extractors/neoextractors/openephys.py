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

from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts


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
        The folder path to load the recordings from.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    block_index: int, optional
        If there are several blocks (experiments), specify the block index you want to load.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.
    """
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysRawIO'
    name = "openephyslegacy"

    def __init__(self, folder_path, stream_id=None, stream_name=None, block_index=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id,
                                           stream_name=stream_name,
                                           block_index=block_index,
                                           all_annotations=all_annotations, 
                                           **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path)))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {'dirname': str(folder_path)}
        return neo_kwargs


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
        The folder path to load the recordings from.
    load_sync_channel : bool
        If False (default) and a SYNC channel is present (e.g. Neuropixels), this is not loaded.
        If True, the SYNC channel is loaded and can be accessed in the analog signals.
    experiment_name: str, list, or None
        If multiple experiments are available, this argument allows users to select one
        or more experiments. If None, all experiements are loaded as blocks.
        E.g. `experiment_names="experiment2"`, `experiment_names=["experiment1", "experiment2"]`
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    block_index: int, optional
        If there are several blocks (experiments), specify the block index you want to load.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.

    """
    mode = 'folder'
    NeoRawIOClass = 'OpenEphysBinaryRawIO'
    name = "openephys"
    has_default_locations = True

    def __init__(self, folder_path, load_sync_channel=False, experiment_names=None,
                 stream_id=None, stream_name=None, block_index=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, load_sync_channel, experiment_names)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id,
                                           stream_name=stream_name,
                                           block_index=block_index,
                                           all_annotations=all_annotations, 
                                           **neo_kwargs)
        # get streams to find correct probe
        stream_names, stream_ids = self.get_streams(folder_path, experiment_names)
        if stream_name is None and stream_id is None:
            stream_name = stream_names[0]
        elif stream_name is None:
            stream_name = stream_names[stream_ids.index(stream_id)]

        # do not load probe for NIDQ stream
        if "NI-DAQmx" not in stream_name:
            # find settings file
            if "#" in stream_name:
                record_node = stream_name.split("#")[0]
            else:
                record_node = ''
            exp_ids = sorted(list(self.neo_reader.folder_structure[record_node]["experiments"].keys()))
            if block_index is None:
                exp_id = exp_ids[0]
            else:
                exp_id = exp_ids[block_index]
            settings_file = self.neo_reader.folder_structure[record_node]["experiments"][exp_id]["settings_file"]

            if Path(settings_file).is_file():
                probe = pi.read_openephys(settings_file=settings_file,
                                          stream_name=stream_name, raise_error=False)
            else:
                probe = None

            if probe is not None:
                self = self.set_probe(probe, in_place=True)
                probe_name = probe.annotations["probe_name"]
                # load num_channels_per_adc depending on probe type
                if "2.0" in probe_name:
                    num_channels_per_adc = 16
                else:
                    num_channels_per_adc = 12
                sample_shifts = get_neuropixels_sample_shifts(self.get_num_channels(), num_channels_per_adc)
                self.set_property("inter_sample_shift", sample_shifts)
        self._kwargs.update(dict(folder_path=str(folder_path),
                                 load_sync_channel=load_sync_channel,
                                 experiment_names=experiment_names))


    @classmethod
    def map_to_neo_kwargs(cls, folder_path, load_sync_channel=False, experiment_names=None):
        neo_kwargs = {'dirname': str(folder_path),
                      'load_sync_channel': load_sync_channel,
                      'experiment_names': experiment_names}
        return neo_kwargs


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
    name = "openephys"

    def __init__(self, folder_path):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseEventExtractor.__init__(self, **neo_kwargs)

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {'dirname': str(folder_path)}
        return neo_kwargs


def read_openephys(folder_path, **kwargs):
    """
    Read 'legacy' or 'binary' Open Ephys formats

    Parameters
    ----------
    folder_path: str or Path
        Path to openephys folder
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    block_index: int, optional
        If there are several blocks (experiments), specify the block index you want to load.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.

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
