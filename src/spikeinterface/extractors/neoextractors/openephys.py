"""
There are two extractors for data saved by the Open Ephys GUI

  * OpenEphysLegacyRecordingExtractor: reads the original "Open Ephys" data format
  * OpenEphysBinaryRecordingExtractor: reads the new default "Binary" format

See https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/index.html
for more info.
"""

from __future__ import annotations


from pathlib import Path

import numpy as np
import warnings

import probeinterface

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor, NeoBaseEventExtractor

from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts


def drop_invalid_neo_arguments_for_version_0_12_0(neo_kwargs):
    from packaging.version import Version
    from importlib.metadata import version as lib_version

    # Temporary function until neo version 0.13.0 is released
    neo_version = lib_version("neo")
    # The possibility of ignoring timestamps errors is not present in neo <= 0.12.0
    if Version(neo_version) <= Version("0.12.0"):
        neo_kwargs.pop("ignore_timestamps_errors")

    return neo_kwargs


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
        The folder path to load the recordings from
    stream_id: str, default: None
        If there are several streams, specify the stream id you want to load
    stream_name: str, default: None
        If there are several streams, specify the stream name you want to load
    block_index: int, default: None
        If there are several blocks (experiments), specify the block index you want to load
    all_annotations: bool, default: False
        Load exhaustively all annotation from neo
    ignore_timestamps_errors: None
        Deprecated keyword argument. This is now ignored.
        neo.OpenEphysRawIO is now handling gaps directly but makes the read slower.
    """

    mode = "folder"
    NeoRawIOClass = "OpenEphysRawIO"
    name = "openephyslegacy"

    def __init__(
        self,
        folder_path,
        stream_id=None,
        stream_name=None,
        block_index=None,
        all_annotations=False,
        ignore_timestamps_errors=None,
    ):
        if ignore_timestamps_errors is not None:
            warnings.warn(
                "OpenEphysLegacyRecordingExtractor: ignore_timestamps_errors is deprecated and is ignored",
                DeprecationWarning,
                stacklevel=2,
            )
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            block_index=block_index,
            all_annotations=all_annotations,
            **neo_kwargs,
        )
        self._kwargs.update(dict(folder_path=str(Path(folder_path).absolute())))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        neo_kwargs = drop_invalid_neo_arguments_for_version_0_12_0(neo_kwargs)
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
        The folder path to the root folder (containing the record node folders)
    load_sync_channel : bool, default: False
        If False (default) and a SYNC channel is present (e.g. Neuropixels), this is not loaded
        If True, the SYNC channel is loaded and can be accessed in the analog signals.
    load_sync_timestamps : bool, default: False
        If True, the synchronized_timestamps are loaded and set as times to the recording.
        If False (default), only the t_start and sampling rate are set, and timestamps are assumed
        to be uniform and linearly increasing
    experiment_names: str, list, or None, default: None
        If multiple experiments are available, this argument allows users to select one
        or more experiments. If None, all experiements are loaded as blocks.
        E.g. `experiment_names="experiment2"`, `experiment_names=["experiment1", "experiment2"]`
    stream_id: str, default: None
        If there are several streams, specify the stream id you want to load
    stream_name: str, default: None
        If there are several streams, specify the stream name you want to load
    block_index: int, default: None
        If there are several blocks (experiments), specify the block index you want to load
    all_annotations: bool, default: False
        Load exhaustively all annotation from neo

    """

    mode = "folder"
    NeoRawIOClass = "OpenEphysBinaryRawIO"
    name = "openephys"

    def __init__(
        self,
        folder_path,
        load_sync_channel=False,
        load_sync_timestamps=False,
        experiment_names=None,
        stream_id=None,
        stream_name=None,
        block_index=None,
        all_annotations=False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, load_sync_channel, experiment_names)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            block_index=block_index,
            all_annotations=all_annotations,
            **neo_kwargs,
        )
        # get streams to find correct probe
        stream_names, stream_ids = self.get_streams(folder_path, experiment_names)
        if stream_name is None and stream_id is None:
            stream_name = stream_names[0]
        elif stream_name is None:
            stream_name = stream_names[stream_ids.index(stream_id)]

        # find settings file
        if "#" in stream_name:
            record_node, oe_stream = stream_name.split("#")
        else:
            record_node = ""
            oe_stream = stream_name
        exp_ids = sorted(list(self.neo_reader.folder_structure[record_node]["experiments"].keys()))
        if block_index is None:
            exp_id = exp_ids[0]
        else:
            exp_id = exp_ids[block_index]
        rec_ids = sorted(
            list(self.neo_reader.folder_structure[record_node]["experiments"][exp_id]["recordings"].keys())
        )

        # do not load probe for NIDQ stream or if load_sync_channel is True
        if "NI-DAQmx" not in stream_name and not load_sync_channel:
            settings_file = self.neo_reader.folder_structure[record_node]["experiments"][exp_id]["settings_file"]

            if Path(settings_file).is_file():
                probe = probeinterface.read_openephys(
                    settings_file=settings_file, stream_name=stream_name, raise_error=False
                )
            else:
                probe = None

            if probe is not None:
                if probe.shank_ids is not None:
                    self.set_probe(probe, in_place=True, group_mode="by_shank")
                else:
                    self.set_probe(probe, in_place=True)

                # this handles a breaking change in probeinterface after v0.2.18
                # in the new version, the Neuropixels model name is stored in the "model_name" annotation,
                # rather than in the "probe_name" annotation
                model_name = probe.annotations.get("model_name", None)
                if model_name is None:
                    model_name = probe.annotations["probe_name"]

                # load num_channels_per_adc depending on probe type
                if "2.0" in model_name:
                    num_channels_per_adc = 16
                    num_cycles_in_adc = 16
                    total_channels = 384
                else:  # NP1.0
                    num_channels_per_adc = 12
                    num_cycles_in_adc = 13 if "AP" in stream_name else 12
                    total_channels = 384

                # sample_shifts is generated from total channels (384) channels
                # when only some channels are saved we need to slice this vector (like we do for the probe)
                sample_shifts = get_neuropixels_sample_shifts(total_channels, num_channels_per_adc, num_cycles_in_adc)
                if self.get_num_channels() != total_channels:
                    # need slice because not all channel are saved
                    chans = probeinterface.get_saved_channel_indices_from_openephys_settings(settings_file, oe_stream)
                    # lets clip to 384 because this contains also the synchro channel
                    chans = chans[chans < total_channels]
                    sample_shifts = sample_shifts[chans]
                self.set_property("inter_sample_shift", sample_shifts)

        # load synchronized timestamps and set_times to recording
        recording_folder = Path(folder_path) / record_node
        stream_folders = []
        for segment_index, rec_id in enumerate(rec_ids):
            stream_folder = recording_folder / f"experiment{exp_id}" / f"recording{rec_id}" / "continuous" / oe_stream
            stream_folders.append(stream_folder)
            if load_sync_timestamps:
                if (stream_folder / "sample_numbers.npy").is_file():
                    # OE version>=v0.6
                    sync_times = np.load(stream_folder / "timestamps.npy")
                elif (stream_folder / "synchronized_timestamps.npy").is_file():
                    # version<v0.6
                    sync_times = np.load(stream_folder / "synchronized_timestamps.npy")
                else:
                    sync_times = None
                try:
                    self.set_times(times=sync_times, segment_index=segment_index, with_warning=False)
                except:
                    warnings.warn(f"Could not load synchronized timestamps for {stream_name}")

        self._stream_folders = stream_folders

        self._kwargs.update(
            dict(
                folder_path=str(Path(folder_path).absolute()),
                load_sync_channel=load_sync_channel,
                load_sync_timestamps=load_sync_timestamps,
                experiment_names=experiment_names,
            )
        )

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, load_sync_channel=False, experiment_names=None):
        neo_kwargs = {
            "dirname": str(folder_path),
            "load_sync_channel": load_sync_channel,
            "experiment_names": experiment_names,
        }
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

    mode = "folder"
    NeoRawIOClass = "OpenEphysBinaryRawIO"
    name = "openephys"

    def __init__(self, folder_path, block_index=None):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseEventExtractor.__init__(self, block_index=block_index, **neo_kwargs)

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        return neo_kwargs


def read_openephys(folder_path, **kwargs):
    """
    Read "legacy" or "binary" Open Ephys formats

    Parameters
    ----------
    folder_path: str or Path
        Path to openephys folder
    stream_id: str, default: None
        If there are several streams, specify the stream id you want to load
    stream_name: str, default: None
        If there are several streams, specify the stream name you want to load
    block_index: int, default: None
        If there are several blocks (experiments), specify the block index you want to load
    all_annotations: bool, default: False
        Load exhaustively all annotation from neo
    load_sync_channel : bool, default: False
        If False (default) and a SYNC channel is present (e.g. Neuropixels), this is not loaded.
        If True, the SYNC channel is loaded and can be accessed in the analog signals.
        For open ephsy binary format only
    load_sync_timestamps : bool, default: False
        If True, the synchronized_timestamps are loaded and set as times to the recording.
        If False (default), only the t_start and sampling rate are set, and timestamps are assumed
        to be uniform and linearly increasing.
        For open ephsy binary format only
    experiment_names: str, list, or None, default: None
        If multiple experiments are available, this argument allows users to select one
        or more experiments. If None, all experiements are loaded as blocks.
        E.g. `experiment_names="experiment2"`, `experiment_names=["experiment1", "experiment2"]`
        For open ephsy binary format only
    ignore_timestamps_errors: bool, default: False
        Ignore the discontinuous timestamps errors in neo
        For open ephsy legacy format only


    Returns
    -------
    recording: OpenEphysLegacyRecordingExtractor or OpenEphysBinaryExtractor
    """
    # auto guess format
    files = [f for f in Path(folder_path).iterdir()]
    if np.any([".continuous" in f.name and f.is_file() for f in files]):
        # format = 'legacy'
        recording = OpenEphysLegacyRecordingExtractor(folder_path, **kwargs)
    else:
        # format = 'binary'
        recording = OpenEphysBinaryRecordingExtractor(folder_path, **kwargs)
    return recording


def read_openephys_event(folder_path, block_index=None):
    """
    Read Open Ephys events from "binary" format.

    Parameters
    ----------
    folder_path: str or Path
        Path to openephys folder
    block_index: int, default: None
        If there are several blocks (experiments), specify the block index you want to load.

    Returns
    -------
    event: OpenEphysBinaryEventExtractor
    """
    # auto guess format
    files = [str(f) for f in Path(folder_path).iterdir()]
    if np.any([f.startswith("Continuous") for f in files]):
        raise Exception("Events can be read only from 'binary' format")
    else:
        # format = 'binary'
        event = OpenEphysBinaryEventExtractor(folder_path, block_index=block_index)
    return event
