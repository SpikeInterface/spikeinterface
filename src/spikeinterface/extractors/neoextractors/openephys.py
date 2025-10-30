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

from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts_from_probe
from spikeinterface.extractors.neoextractors.neobaseextractor import NeoBaseRecordingExtractor, NeoBaseEventExtractor


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
    folder_path : str
        The folder path to load the recordings from
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load
    block_index : int, default: None
        If there are several blocks (experiments), specify the block index you want to load
    all_annotations : bool, default: False
        Load exhaustively all annotation from neo
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.
    ignore_timestamps_errors : None
        Deprecated keyword argument. This is now ignored.
        neo.OpenEphysRawIO is now handling gaps directly but makes the read slower.
    """

    NeoRawIOClass = "OpenEphysRawIO"

    def __init__(
        self,
        folder_path,
        stream_id=None,
        stream_name=None,
        block_index=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
        ignore_timestamps_errors: bool = None,
    ):
        if ignore_timestamps_errors is not None:
            dep_msg = "OpenEphysLegacyRecordingExtractor: `ignore_timestamps_errors` is deprecated. It will be removed in version 0.104.0 and is currently ignored"
            warnings.warn(
                dep_msg,
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
            use_names_as_ids=use_names_as_ids,
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
    Class for reading data saved by the Open Ephys GUI in "binary" format.

    This extractor reads Open Ephys binary format data, which organizes recordings in a hierarchical
    structure: Record Nodes (hardware devices) contain Experiments (experimental sessions or groupings)
    which contain Recordings (individual recording sessions). Each recording contains continuous
    signal streams (.dat files) and event streams.

    https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/Binary-format.html

    Based on neo.rawio.OpenEphysBinaryRawIO

    Parameters
    ----------
    folder_path : str or Path
        Path to the Open Ephys data directory. Can point to:
        - Root folder containing Record Node folders (recommended for multi-node recordings)
        - Specific Record Node folder (e.g., "Record Node 102")
        - Specific experiment folder (e.g., "experiment1")
        - Specific recording folder (e.g., "recording1")
        The reader will automatically detect the directory level and parse accordingly.
    experiment_name : str or None, default: None
        Name of the experiment to load (e.g., "experiment1", "experiment2").
        If multiple experiments are available and neither experiment_name nor block_index is specified,
        an error will be raised listing all available experiments.
        Use the get_available_experiments() class method to discover available experiments.
        Note: Only one experiment can be loaded at a time in SpikeInterface.
        Cannot be used together with block_index.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load
    block_index : int or None, default: None
        Alternative way to specify which experiment to load using a zero-based index.
        block_index=0 corresponds to experiment1, block_index=1 to experiment2, etc.
        Cannot be used together with experiment_name.
    load_sync_channel : bool, default: False
        **DEPRECATED: Use stream_name or stream_id to load sync streams. Will be removed in version 0.104.0**
        If False (default) and a SYNC channel is present (e.g., Neuropixels), this is not loaded.
        If True, the SYNC channel is loaded and can be accessed in the analog signals.
    load_sync_timestamps : bool, default: False
        If True, the synchronized_timestamps are loaded and set as times to the recording.
        If False (default), only the t_start and sampling rate are set, and timestamps are assumed
        to be uniform and linearly increasing
    experiment_names : str, list, or None, default: None
        **DEPRECATED: Use experiment_name instead. Will be removed in version 0.105.0**
        This parameter was designed for Neo's multi-block loading, but SpikeInterface only loads
        one block at a time. Use experiment_name to select a single experiment.
    all_annotations : bool, default: False
        Load exhaustively all annotation from neo

    Notes
    -----
    Open Ephys Binary Format Structure:
        folder_path/
        ├── Record Node 102/              # Recording hardware node
        │   ├── settings.xml              # Settings for the first experiment
        │   ├── settings_2.xml            # Settings for experiment 2
        │   ├── experiment1/              # Experiment folder
        │   │   ├── recording1/           # Recording session (SpikeInterface segment)
        │   │   │   ├── structure.oebin   # JSON metadata file
        │   │   │   ├── continuous/       # Signal streams
        │   │   │   │   └── Neuropix-PXI-100.ProbeA-AP/
        │   │   │   │       ├── continuous.dat
        │   │   │   │       └── timestamps.npy
        │   │   │   └── events/           # Event streams
        │   │   └── recording2/           # Additional recording (additional segment)
        │   └── experiment2/              # Different experiment
        └── Record Node 103/              # Second hardware node (if present)

    Open Ephys to SpikeInterface Mapping:
        - **Experiment** (experiment1, experiment2, ...)
          → One SpikeInterface Recording object (select with experiment_name parameter)
        - **Recording** (recording1, recording2, ...) within an experiment
          → Segments within the Recording object (access via get_num_segments())
        - **Continuous stream** (AP_band, LF_band, ...)
          → The signal data loaded into the Recording (select with stream_name/stream_id)

    Common Use Cases:
        1. Single experiment dataset:
           Simply specify folder_path, experiment will be auto-selected

        2. Multi-experiment dataset:
           Use get_available_experiments() to discover, then select with experiment_name

        3. Multi-stream recording (e.g., Neuropixels AP + LF):
           Use stream_name or stream_id to select which stream to load

        4. Multi-recording experiment:
           All recordings within an experiment are loaded as segments automatically

        5. Multi-node recording:
           Stream names will be prefixed with node name (e.g., "Record Node 102#AP")

    See Also
    --------
    get_available_experiments : Discover available experiments in a dataset
    get_streams : Discover available streams in a dataset

    """

    NeoRawIOClass = "OpenEphysBinaryRawIO"

    @classmethod
    def get_available_experiments(cls, folder_path):
        """
        Get list of available experiment names in an Open Ephys binary folder.

        Parameters
        ----------
        folder_path : str or Path
            Path to the Open Ephys data directory

        Returns
        -------
        experiment_names : list of str
            List of available experiment names (e.g., ["experiment1", "experiment2"])
        """
        from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO

        _, possible_experiments = OpenEphysBinaryRawIO._parse_folder_structure(str(folder_path), experiment_names=None)
        return possible_experiments

    def __init__(
        self,
        folder_path: str | Path,
        experiment_name: str | None = None,
        stream_id: str = None,
        stream_name: str = None,
        block_index: int = None,
        load_sync_channel: bool = False,
        load_sync_timestamps: bool = False,
        experiment_names: str | list | None = None,
        all_annotations: bool = False,
    ):
        # Handle experiment_names deprecation
        if experiment_names is not None:
            warnings.warn(
                "OpenEphysBinaryRecordingExtractor: 'experiment_names' is deprecated and will be removed in version 0.105.0. "
                "Use 'experiment_name' instead to select a single experiment (e.g., experiment_name='experiment2').",
                FutureWarning,
                stacklevel=2,
            )

        # Handle experiment_name and block_index parameters
        if experiment_name is not None and block_index is not None:
            raise ValueError(
                "OpenEphysBinaryRecordingExtractor: Cannot specify both 'experiment_name' and 'block_index'. "
                "Please use either 'experiment_name' or 'block_index', but not both."
            )

        # Convert experiment_name to experiment_names for Neo
        # When using experiment_name, Neo will filter to only that experiment, making it block_index=0
        # experiment_name takes precedence over experiment_names
        experiment_names_for_neo = experiment_names  # Use deprecated parameter if provided
        if experiment_name is not None:
            # experiment_name overrides experiment_names
            experiment_names_for_neo = [experiment_name]
            # Validate that the experiment exists
            available_experiments = self.get_available_experiments(folder_path)
            if experiment_name not in available_experiments:
                raise ValueError(
                    f"OpenEphysBinaryRecordingExtractor: experiment_name '{experiment_name}' not found. "
                    f"Available experiments: {available_experiments}"
                )
            experiment_names_for_neo = [experiment_name]
            # When filtering to a single experiment, it becomes block 0
            block_index = 0
        elif block_index is None and experiment_names_for_neo is None:
            # If neither experiment_name, experiment_names, nor block_index is provided,
            # check for multiple experiments and provide a helpful error message
            available_experiments = self.get_available_experiments(folder_path)
            if len(available_experiments) > 1:
                raise ValueError(
                    f"OpenEphysBinaryRecordingExtractor: Multiple experiments found: {available_experiments}. "
                    f"Please specify which experiment to load using the 'experiment_name' parameter. "
                    f"Example: experiment_name='{available_experiments[0]}'"
                )
            # Single experiment: no filtering needed, let base class handle it
            block_index = None

        if load_sync_channel:
            warning_message = (
                "OpenEphysBinaryRecordingExtractor: `load_sync_channel` is deprecated and will "
                "be removed in version 0.104, use the `stream_name` or `stream_id` to load the sync stream if needed"
            )
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)

        stream_is_not_specified = stream_name is None and stream_id is None
        if stream_is_not_specified:
            available_stream_names, _ = self.get_streams(folder_path, load_sync_channel, experiment_names_for_neo)

            # Auto-select neural data stream when there are exactly two streams (neural + sync)
            # and no stream was explicitly specified
            if len(available_stream_names) == 2:
                has_sync_stream = any("SYNC" in stream for stream in available_stream_names)
                if has_sync_stream:
                    neural_stream_name = next(stream for stream in available_stream_names if "SYNC" not in stream)
                    stream_name = neural_stream_name

        neo_kwargs = self.map_to_neo_kwargs(folder_path, load_sync_channel, experiment_names_for_neo)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            block_index=block_index,
            all_annotations=all_annotations,
            **neo_kwargs,
        )

        stream_is_sync = "SYNC" in self.stream_name
        if not stream_is_sync:
            # get streams to find correct probe
            stream_names, stream_ids = self.get_streams(folder_path, load_sync_channel, experiment_names_for_neo)
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
                    # get inter-sample shifts based on the probe information and mux channels
                    sample_shifts = get_neuropixels_sample_shifts_from_probe(probe, stream_name=self.stream_name)
                    if sample_shifts is not None:
                        num_readout_channels = probe.annotations.get("num_readout_channels")
                        if self.get_num_channels() != num_readout_channels:
                            # need slice because not all channels are saved
                            chans = probeinterface.get_saved_channel_indices_from_openephys_settings(
                                settings_file, oe_stream
                            )
                            # lets clip to num_readout_channels because this contains also the synchro channel
                            if chans is not None:
                                chans = chans[chans < num_readout_channels]
                                sample_shifts = sample_shifts[chans]
                        self.set_property("inter_sample_shift", sample_shifts)

            # load synchronized timestamps and set_times to recording
            recording_folder = Path(folder_path) / record_node
            stream_folders = []
            for segment_index, rec_id in enumerate(rec_ids):
                stream_folder = (
                    recording_folder / f"experiment{exp_id}" / f"recording{rec_id}" / "continuous" / oe_stream
                )
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

            self.annotate(experiment_name=f"experiment{exp_id}")
            self._stream_folders = stream_folders

        self._kwargs.update(
            dict(
                folder_path=str(Path(folder_path).absolute()),
                experiment_name=experiment_name,
                load_sync_channel=load_sync_channel,
                load_sync_timestamps=load_sync_timestamps,
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
    folder_path : str
        Path to the Open Ephys data directory
    experiment_name : str or None, default: None
        Name of the experiment to load (e.g., "experiment1", "experiment2").
        Cannot be used together with block_index.
    block_index : int or None, default: None
        Alternative way to specify which experiment to load using a zero-based index.
        Cannot be used together with experiment_name.

    """

    NeoRawIOClass = "OpenEphysBinaryRawIO"

    @classmethod
    def get_available_experiments(cls, folder_path):
        """
        Get list of available experiment names in an Open Ephys binary folder.

        Parameters
        ----------
        folder_path : str or Path
            Path to the Open Ephys data directory

        Returns
        -------
        experiment_names : list of str
            List of available experiment names (e.g., ["experiment1", "experiment2"])
        """
        from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO

        _, possible_experiments = OpenEphysBinaryRawIO._parse_folder_structure(str(folder_path), experiment_names=None)
        return possible_experiments

    def __init__(self, folder_path, experiment_name=None, block_index=None):
        # Handle experiment_name and block_index parameters
        if experiment_name is not None and block_index is not None:
            raise ValueError(
                "OpenEphysBinaryEventExtractor: Cannot specify both 'experiment_name' and 'block_index'. "
                "Please use either 'experiment_name' or 'block_index', but not both."
            )

        # Convert experiment_name to experiment_names for Neo
        experiment_names_for_neo = None
        if experiment_name is not None:
            # Validate that the experiment exists
            available_experiments = self.get_available_experiments(folder_path)
            if experiment_name not in available_experiments:
                raise ValueError(
                    f"OpenEphysBinaryEventExtractor: experiment_name '{experiment_name}' not found. "
                    f"Available experiments: {available_experiments}"
                )
            experiment_names_for_neo = [experiment_name]
            # When filtering to a single experiment, it becomes block 0
            block_index = 0
        elif block_index is None and experiment_names_for_neo is None:
            # If neither experiment_name nor block_index is provided,
            # check for multiple experiments and provide a helpful error message
            available_experiments = self.get_available_experiments(folder_path)
            if len(available_experiments) > 1:
                raise ValueError(
                    f"OpenEphysBinaryEventExtractor: Multiple experiments found: {available_experiments}. "
                    f"Please specify which experiment to load using the 'experiment_name' parameter. "
                    f"Example: experiment_name='{available_experiments[0]}'"
                )
            # Single experiment: no filtering needed
            block_index = None

        neo_kwargs = self.map_to_neo_kwargs(folder_path, experiment_names_for_neo)
        NeoBaseEventExtractor.__init__(self, block_index=block_index, **neo_kwargs)

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, experiment_names=None):
        neo_kwargs = {"dirname": str(folder_path), "experiment_names": experiment_names}
        return neo_kwargs


def read_openephys(folder_path, **kwargs):
    """
    Read Open Ephys folder (in "binary" or "open ephys legacy" format).

    Parameters
    ----------
    folder_path : str or Path
        Path to openephys folder
    experiment_name : str, default: None
        Name of the experiment to load (e.g., "experiment1", "experiment2").
        For open ephys binary format only. Cannot be used together with block_index.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load
    block_index : int, default: None
        Alternative way to specify which experiment to load using a zero-based index.
        If there are several blocks (experiments), specify the block index you want to load.
        Cannot be used together with experiment_name.
    all_annotations : bool, default: False
        Load exhaustively all annotation from neo
    load_sync_channel : bool, default: False
        **DEPRECATED: Use stream_name or stream_id to load sync streams**
        If False (default) and a SYNC channel is present (e.g. Neuropixels), this is not loaded.
        If True, the SYNC channel is loaded and can be accessed in the analog signals.
        For open ephys binary format only
    load_sync_timestamps : bool, default: False
        If True, the synchronized_timestamps are loaded and set as times to the recording.
        If False (default), only the t_start and sampling rate are set, and timestamps are assumed
        to be uniform and linearly increasing.
        For open ephys binary format only
    ignore_timestamps_errors : bool, default: False
        Ignore the discontinuous timestamps errors in neo
        For open ephys legacy format only


    Returns
    -------
    recording : OpenEphysLegacyRecordingExtractor or OpenEphysBinaryRecordingExtractor
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


def read_openephys_event(folder_path, experiment_name=None, block_index=None):
    """
    Read Open Ephys events from "binary" format.

    Parameters
    ----------
    folder_path : str or Path
        Path to openephys folder
    experiment_name : str or None, default: None
        Name of the experiment to load (e.g., "experiment1", "experiment2").
        Cannot be used together with block_index.
    block_index : int, default: None
        Alternative way to specify which experiment to load using a zero-based index.
        If there are several blocks (experiments), specify the block index you want to load.
        Cannot be used together with experiment_name.

    Returns
    -------
    event : OpenEphysBinaryEventExtractor
    """
    # auto guess format
    files = [str(f) for f in Path(folder_path).iterdir()]
    if np.any([f.startswith("Continuous") for f in files]):
        raise Exception("Events can be read only from 'binary' format")
    else:
        # format = 'binary'
        event = OpenEphysBinaryEventExtractor(folder_path, experiment_name=experiment_name, block_index=block_index)
    return event
