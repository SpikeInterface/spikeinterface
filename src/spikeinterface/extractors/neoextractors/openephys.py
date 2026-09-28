"""
There are two extractors for data saved by the Open Ephys GUI

  * OpenEphysLegacyRecordingExtractor: reads the original "Open Ephys" data format
  * OpenEphysBinaryRecordingExtractor: reads the new default "Binary" format

See https://open-ephys.github.io/gui-docs/User-Manual/Recording-data/index.html
for more info.
"""

import importlib.util
from pathlib import Path

import numpy as np
import warnings

import probeinterface

from spikeinterface.extractors.neuropixels_utils import (
    get_neuropixels_sample_shifts_from_probe,
    compute_saturation_threshold_from_probe,
)
from spikeinterface.extractors.neoextractors.neobaseextractor import NeoBaseRecordingExtractor, NeoBaseEventExtractor

from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.core import BaseRecording, BaseRecordingSegment


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
    ):
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
    load_sync_timestamps : bool, default: False
        If True, the synchronized_timestamps are loaded and set as times to the recording.
        If False (default), only the t_start and sampling rate are set, and timestamps are assumed
        to be uniform and linearly increasing
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
        load_sync_timestamps: bool = False,
        all_annotations: bool = False,
    ):
        folder_path = Path(folder_path)

        # Handle experiment_name and block_index parameters
        if experiment_name is not None and block_index is not None:
            raise ValueError(
                "OpenEphysBinaryRecordingExtractor: Cannot specify both 'experiment_name' and 'block_index'. "
                "Please use either 'experiment_name' or 'block_index', but not both."
            )

        # Convert experiment_name to experiment_names for Neo
        # When using experiment_name, Neo will filter to only that experiment, making it block_index=0
        # experiment_name takes precedence over experiment_names
        experiment_names_for_neo = None  # No longer using the deprecated parameter
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

        stream_is_not_specified = stream_name is None and stream_id is None
        if stream_is_not_specified:
            available_stream_names, _ = self.get_streams(folder_path, experiment_names_for_neo)

            # Auto-select neural data stream when there are exactly two streams (neural + sync)
            # and no stream was explicitly specified
            if len(available_stream_names) == 2:
                has_sync_stream = any("SYNC" in stream for stream in available_stream_names)
                if has_sync_stream:
                    neural_stream_name = next(stream for stream in available_stream_names if "SYNC" not in stream)
                    stream_name = neural_stream_name

        neo_kwargs = self.map_to_neo_kwargs(folder_path, experiment_names_for_neo)
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
            stream_names, stream_ids = self.get_streams(folder_path, experiment_names_for_neo)
            if stream_name is None and stream_id is None:
                stream_name = stream_names[0]
            elif stream_name is None:
                stream_name = stream_names[stream_ids.index(stream_id)]

            # find settings file
            if "#" in stream_name:
                record_node, oe_stream_name = stream_name.split("#")
            else:
                record_node = ""
                oe_stream_name = stream_name
            node_structure = self.neo_reader.folder_structure[record_node]
            exp_ids = sorted(list(node_structure["experiments"].keys()))
            if block_index is None:
                exp_id = exp_ids[0]
            else:
                exp_id = exp_ids[block_index]
            rec_ids = sorted(list(node_structure["experiments"][exp_id]["recordings"].keys()))

            # do not load probe for NIDQ stream
            if "NI-DAQmx" not in stream_name:
                settings_file = node_structure["experiments"][exp_id]["settings_file"]

                if Path(settings_file).is_file() and probeinterface.has_neuropixels_probes(
                    settings_file, stream_name=oe_stream_name
                ):
                    probe = probeinterface.read_openephys_neuropixels(
                        settings_file=settings_file, stream_name=oe_stream_name
                    )
                    if probe.shank_ids is not None:
                        self.set_probe(probe, group_mode="by_shank")
                    else:
                        self.set_probe(probe)
                    # get inter-sample shifts based on the probe information and mux channels
                    sample_shifts = get_neuropixels_sample_shifts_from_probe(probe)
                    if sample_shifts is not None:
                        self.set_property("inter_sample_shift", sample_shifts)

                    # add saturation levels if available
                    saturation_threshold_uV = compute_saturation_threshold_from_probe(probe, oe_stream_name)
                    if saturation_threshold_uV is not None:
                        self.annotate(saturation_threshold_uV=saturation_threshold_uV)

            # folder_path can point to different levels of the OE folder structure
            # (root, record node, experiment, or recording). We need to find the root folder
            # in order to load the sync timestamps and set them as times to the recording.
            if record_node in folder_path.parts:
                root_index = len(folder_path.parts) - folder_path.parts.index(record_node) - 1
                root_folder = folder_path.parents[root_index]
            else:
                root_folder = folder_path
            recording_folder = root_folder / record_node
            stream_folders = []
            for segment_index, rec_id in enumerate(rec_ids):
                stream_folder = (
                    recording_folder / f"experiment{exp_id}" / f"recording{rec_id}" / "continuous" / oe_stream_name
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
                load_sync_timestamps=load_sync_timestamps,
            )
        )

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, experiment_names=None):
        neo_kwargs = {
            "dirname": str(folder_path),
            "experiment_names": experiment_names,
        }
        return neo_kwargs

    @classmethod
    def _handle_kwargs_backward_compatibility(cls, old_kwargs, full_dict):
        if "load_sync_channel" in old_kwargs:
            new_kwargs = old_kwargs.copy()
            new_kwargs.pop("load_sync_channel")
        else:
            new_kwargs = old_kwargs
        return new_kwargs


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


class OpenEphysArrowRecordingSegment(BaseRecordingSegment):
    def __init__(self, filepath, channel_ids, batch_len, **time_kwargs):
        BaseRecordingSegment.__init__(self, **time_kwargs)

        from pyarrow import memory_map
        from pyarrow.ipc import RecordBatchFileReader

        self._source = memory_map(filepath, "r")
        self._reader = RecordBatchFileReader(self._source)
        self.batch_len = batch_len

        self._all_channel_ids = channel_ids

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return 18_000_000

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list[int | str] | None = None,
    ) -> np.ndarray:
        if channel_indices is None:
            channel_ids = list(self._all_channel_ids)
        else:
            channel_ids = list(self._all_channel_ids[channel_indices])

        import pyarrow as pa

        # Arrow saves data in "batch"es, in the time dimension, which we can
        # load individually. We need to figure out which batches our requested
        # samples are in.

        batch_size = self.batch_len
        first_batch_idx = start_frame // batch_size
        last_batch_idx = (end_frame - 1) // batch_size

        # This is super easy if our samples are in a single batch
        if first_batch_idx == last_batch_idx:
            batch = self._reader.get_batch(first_batch_idx)
            local_start = start_frame % batch_size
            sliced = batch.slice(local_start, end_frame - start_frame)
            return np.column_stack([sliced.column(c).to_numpy(zero_copy_only=False) for c in channel_ids])

        # Otherwise, we find all batches, then grab the data
        slices = []
        for b_idx in range(first_batch_idx, last_batch_idx + 1):
            batch = self._reader.get_batch(b_idx)
            b_start = b_idx * batch_size

            local_start = max(0, start_frame - b_start)
            local_end = min(batch.num_rows, end_frame - b_start)

            slices.append(batch.slice(local_start, local_end - local_start).select(channel_ids))

        table = pa.Table.from_batches(slices)
        return np.column_stack([table[c].to_numpy(zero_copy_only=False) for c in channel_ids])


class OpenEphysArrowRecording(BaseRecording):
    """
    Recording class for the openephys arrow format, from ___

    We assume

    Parameters
    ----------
    file_path : str
        Path to the directory where the zarr array is stored
    sampling_frequency : float
        The sampling frequency
    stream_name : str, default: AmplifierData
        The stream name of the data you want to load. By default, the ephys AP stream is
        called "AmplifierData".
    gain_to_uV : float or array-like, default: None
        The gain to apply to the traces
    offset_to_uV : float or array-like, default: None
        The offset to apply to the traces
    is_filtered : bool or None, default: None
        If True, the recording is assumed to be filtered. If None, is_filtered is not set.
    storage_options : dict or None: None
        Storage options passed to the `zarr.open` function

    Returns
    -------
    recording : ZarrArrayRecording
        The recording Extractor
    """

    def __init__(
        self,
        file_path: str | Path,
        sampling_frequency: float,
        stream_name="AmplifierData",
        gain_to_uV: float | np.ndarray | None = None,
        offset_to_uV: float | np.ndarray | None = None,
        is_filtered: bool | None = None,
    ):
        if importlib.util.find_spec("pyarrow") is None:
            raise ImportError("You need to add `pyarrow` to your environment to open .arrow files")
        else:
            from pyarrow import memory_map
            from pyarrow.ipc import RecordBatchFileReader

        source = memory_map(file_path, "r")
        reader = RecordBatchFileReader(source)

        stream_names = reader.schema.names
        channel_ids = [name for name in stream_names if stream_name in name]

        if len(channel_ids) == 0:
            raise ValueError(f"Cannot find any data with `stream_name` = {stream_name}")

        first_batch = reader.get_batch(0)
        batch_len = first_batch.num_rows

        one_channel_index = stream_names.index(channel_ids[0])

        # Arrow uses it's own DataType. For ints and floats, it converts to numpy dtype without issue
        ephys_type = reader.schema[one_channel_index].type
        numpy_type = np.dtype(str(ephys_type))

        source.close()

        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=numpy_type)

        rec_segment = OpenEphysArrowRecordingSegment(
            file_path, batch_len=batch_len, sampling_frequency=sampling_frequency, channel_ids=np.array(channel_ids)
        )

        self.add_recording_segment(rec_segment)

        if is_filtered is not None:
            self.annotate(is_filtered=is_filtered)

        if gain_to_uV is not None:
            self.set_channel_gains(gain_to_uV)

        if offset_to_uV is not None:
            self.set_channel_offsets(offset_to_uV)

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "sampling_frequency": sampling_frequency,
            "num_channels": len(channel_ids),
            "dtype": numpy_type.str,
            "channel_ids": channel_ids,
            "gain_to_uV": gain_to_uV,
            "offset_to_uV": offset_to_uV,
            "is_filtered": is_filtered,
        }


read_openephys_arrow = define_function_from_class(source_class=OpenEphysArrowRecording, name="read_openephys_arrow")


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
