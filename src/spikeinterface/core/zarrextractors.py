from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import zarr

from probeinterface import ProbeGroup

from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, SpikeVectorSortingSegment, minimum_spike_dtype
from .core_tools import define_function_from_class, check_json
from .job_tools import split_job_kwargs
from .recording_tools import determine_cast_unsigned


class ZarrRecordingExtractor(BaseRecording):
    """
    RecordingExtractor for a zarr format

    Parameters
    ----------
    folder_path : str or Path
        Path to the zarr root folder
    storage_options : dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.

    Returns
    -------
    recording : ZarrRecordingExtractor
        The recording Extractor
    """

    def __init__(self, folder_path: Path | str, storage_options: dict | None = None):

        folder_path, folder_path_kwarg = resolve_zarr_path(folder_path)

        self._root = zarr.open(str(folder_path), mode="r", storage_options=storage_options)

        sampling_frequency = self._root.attrs.get("sampling_frequency", None)
        num_segments = self._root.attrs.get("num_segments", None)
        assert "channel_ids" in self._root.keys(), "'channel_ids' dataset not found!"
        channel_ids = self._root["channel_ids"][:]

        assert sampling_frequency is not None, "'sampling_frequency' attiribute not found!"
        assert num_segments is not None, "'num_segments' attiribute not found!"

        channel_ids = np.array(channel_ids)

        dtype = self._root["traces_seg0"].dtype

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        dtype = np.dtype(dtype)
        t_starts = self._root.get("t_starts", None)

        total_nbytes = 0
        total_nbytes_stored = 0
        cr_by_segment = {}
        for segment_index in range(num_segments):
            trace_name = f"traces_seg{segment_index}"
            assert (
                len(channel_ids) == self._root[trace_name].shape[1]
            ), f"Segment {segment_index} has the wrong number of channels!"

            time_kwargs = {}
            time_vector = self._root.get(f"times_seg{segment_index}", None)
            if time_vector is not None:
                time_kwargs["time_vector"] = time_vector[:]
            else:
                if t_starts is None:
                    t_start = None
                else:
                    t_start = t_starts[segment_index]
                    if np.isnan(t_start):
                        t_start = None
                time_kwargs["t_start"] = t_start
                time_kwargs["sampling_frequency"] = sampling_frequency

            rec_segment = ZarrRecordingSegment(self._root, trace_name, **time_kwargs)

            nbytes_segment = self._root[trace_name].nbytes
            nbytes_stored_segment = self._root[trace_name].nbytes_stored
            cr_by_segment[segment_index] = nbytes_segment / nbytes_stored_segment

            total_nbytes += nbytes_segment
            total_nbytes_stored += nbytes_stored_segment
            self.add_recording_segment(rec_segment)

        # load probe
        probe_dict = self._root.attrs.get("probe", None)
        if probe_dict is not None:
            probegroup = ProbeGroup.from_dict(probe_dict)
            self.set_probegroup(probegroup, in_place=True)

        # load properties
        if "properties" in self._root:
            prop_group = self._root["properties"]
            for key in prop_group.keys():
                values = self._root["properties"][key]
                self.set_property(key, values)

        # load annotations
        annotations = self._root.attrs.get("annotations", None)
        if annotations is not None:
            self.annotate(**annotations)
        # annotate compression ratios
        cr = total_nbytes / total_nbytes_stored
        self.annotate(compression_ratio=cr, compression_ratio_segments=cr_by_segment)

        self._kwargs = {"folder_path": folder_path_kwarg, "storage_options": storage_options}

    @staticmethod
    def write_recording(
        recording: BaseRecording, folder_path: str | Path, storage_options: dict | None = None, **kwargs
    ):
        zarr_root = zarr.open(str(folder_path), mode="w", storage_options=storage_options)
        add_recording_to_zarr_group(recording, zarr_root, **kwargs)


class ZarrRecordingSegment(BaseRecordingSegment):
    def __init__(self, root, dataset_name, **time_kwargs):
        BaseRecordingSegment.__init__(self, **time_kwargs)
        self._timeseries = root[dataset_name]

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self._timeseries.shape[0]

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list[int | str] | None = None,
    ) -> np.ndarray:
        traces = self._timeseries[start_frame:end_frame]
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces


class ZarrSortingExtractor(BaseSorting):
    """
    SortingExtractor for a zarr format

    Parameters
    ----------
    folder_path : str or Path
        Path to the zarr root file
    storage_options : dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.
    zarr_group : str or None, default: None
        Optional zarr group path to load the sorting from. This can be used when the sorting is not stored at the root, but in sub group.
    Returns
    -------
    sorting : ZarrSortingExtractor
        The sorting Extractor
    """

    def __init__(self, folder_path: Path | str, storage_options: dict | None = None, zarr_group: str | None = None):

        folder_path, folder_path_kwarg = resolve_zarr_path(folder_path)

        zarr_root = self._root = zarr.open(str(folder_path), mode="r", storage_options=storage_options)
        if zarr_group is None:
            self._root = zarr_root
        else:
            self._root = zarr_root[zarr_group]

        sampling_frequency = self._root.attrs.get("sampling_frequency", None)
        num_segments = self._root.attrs.get("num_segments", None)
        assert "unit_ids" in self._root.keys(), "'unit_ids' dataset not found!"
        unit_ids = self._root["unit_ids"][:]

        assert sampling_frequency is not None, "'sampling_frequency' attiribute not found!"
        assert num_segments is not None, "'num_segments' attiribute not found!"

        unit_ids = np.array(unit_ids)
        assert "spikes" in self._root.keys(), "'spikes' dataset not found!"
        spikes_group = self._root["spikes"]
        segment_slices_list = spikes_group["segment_slices"][:]

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        spikes = np.zeros(len(spikes_group["sample_index"]), dtype=minimum_spike_dtype)
        spikes["sample_index"] = spikes_group["sample_index"][:]
        spikes["unit_index"] = spikes_group["unit_index"][:]
        for i, (start, end) in enumerate(segment_slices_list):
            spikes["segment_index"][start:end] = i

        for segment_index in range(num_segments):
            soring_segment = SpikeVectorSortingSegment(spikes, segment_index, unit_ids)
            self.add_sorting_segment(soring_segment)

        # load properties
        if "properties" in self._root:
            prop_group = self._root["properties"]
            for key in prop_group.keys():
                values = self._root["properties"][key]
                self.set_property(key, values)

        # load annotations
        annotations = self._root.attrs.get("annotations", None)
        if annotations is not None:
            self.annotate(**annotations)

        self._kwargs = {"folder_path": folder_path_kwarg, "storage_options": storage_options, "zarr_group": zarr_group}

    @staticmethod
    def write_sorting(sorting: BaseSorting, folder_path: str | Path, storage_options: dict | None = None, **kwargs):
        """
        Write a sorting extractor to zarr format.
        """
        zarr_root = zarr.open(str(folder_path), mode="w", storage_options=storage_options)
        add_sorting_to_zarr_group(sorting, zarr_root, **kwargs)


read_zarr_recording = define_function_from_class(source_class=ZarrRecordingExtractor, name="read_zarr_recording")
read_zarr_sorting = define_function_from_class(source_class=ZarrSortingExtractor, name="read_zarr_sorting")


def read_zarr(
    folder_path: str | Path, storage_options: dict | None = None
) -> ZarrRecordingExtractor | ZarrSortingExtractor:
    """
    Read recording or sorting from a zarr format

    Parameters
    ----------
    folder_path : str or Path
        Path to the zarr root file
    storage_options : dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.

    Returns
    -------
    extractor : ZarrExtractor
        The loaded extractor
    """
    # TODO @alessio : we should have something more explicit in our zarr format to tell which object it is.
    # for the futur SortingAnalyzer we will have this 2 fields!!!
    root = zarr.open(str(folder_path), mode="r", storage_options=storage_options)
    if "channel_ids" in root.keys():
        return read_zarr_recording(folder_path, storage_options=storage_options)
    elif "unit_ids" in root.keys():
        return read_zarr_sorting(folder_path, storage_options=storage_options)
    else:
        raise ValueError("Cannot find 'channel_ids' or 'unit_ids' in zarr root. Not a valid SpikeInterface zarr format")


### UTILITY FUNCTIONS ###
def resolve_zarr_path(folder_path: str | Path):
    """
    Resolve a path to a zarr folder.

    Parameters
    ----------
    """
    if str(folder_path).startswith("s3:") or str(folder_path).startswith("gcs:"):
        # cloud location, no need to resolve
        return folder_path, folder_path
    else:
        folder_path = Path(folder_path)
        folder_path_kwarg = str(Path(folder_path).resolve())
        return folder_path, folder_path_kwarg


def get_default_zarr_compressor(clevel: int = 5):
    """
    Return default Zarr compressor object for good preformance in int16
    electrophysiology data.

    cname: zstd (zstandard)
    clevel: 5
    shuffle: BITSHUFFLE

    Parameters
    ----------
    clevel : int, default: 5
        Compression level (higher -> more compressed).
        Minimum 1, maximum 9. By default 5

    Returns
    -------
    Blosc.compressor
        The compressor object that can be used with the save to zarr function
    """
    assert ZarrRecordingExtractor.installed, ZarrRecordingExtractor.installation_mesg
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)


def add_properties_and_annotations(zarr_group: zarr.hierarchy.Group, recording_or_sorting: BaseRecording | BaseSorting):
    # save properties
    prop_group = zarr_group.create_group("properties")
    for key in recording_or_sorting.get_property_keys():
        values = recording_or_sorting.get_property(key)
        if values.dtype.kind == "O":
            warnings.warn(f"Property {key} not saved because it is a python Object type")
            continue
        prop_group.create_dataset(name=key, data=values, compressor=None)

    # save annotations
    zarr_group.attrs["annotations"] = check_json(recording_or_sorting._annotations)


def add_sorting_to_zarr_group(sorting: BaseSorting, zarr_group: zarr.hierarchy.Group, **kwargs):
    """
    Add a sorting extractor to a zarr group.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting extractor object to be added to the zarr group
    zarr_group : zarr.hierarchy.Group
        The zarr group
    kwargs : dict
        Other arguments passed to the zarr compressor
    """
    from numcodecs import Delta

    num_segments = sorting.get_num_segments()
    zarr_group.attrs["sampling_frequency"] = float(sorting.sampling_frequency)
    zarr_group.attrs["num_segments"] = int(num_segments)
    zarr_group.create_dataset(name="unit_ids", data=sorting.unit_ids, compressor=None)

    if "compressor" not in kwargs:
        compressor = get_default_zarr_compressor()

    # save sub fields
    spikes_group = zarr_group.create_group(name="spikes")
    spikes = sorting.to_spike_vector()
    for field in spikes.dtype.fields:
        if field != "segment_index":
            spikes_group.create_dataset(
                name=field,
                data=spikes[field],
                compressor=compressor,
                filters=[Delta(dtype=spikes[field].dtype)],
            )
        else:
            segment_slices = []
            for segment_index in range(num_segments):
                i0, i1 = np.searchsorted(spikes["segment_index"], [segment_index, segment_index + 1])
                segment_slices.append([i0, i1])
            spikes_group.create_dataset(name="segment_slices", data=segment_slices, compressor=None)

    add_properties_and_annotations(zarr_group, sorting)


# Recording
def add_recording_to_zarr_group(
    recording: BaseRecording, zarr_group: zarr.hierarchy.Group, verbose=False, auto_cast_uint=True, dtype=None, **kwargs
):
    zarr_kwargs, job_kwargs = split_job_kwargs(kwargs)

    if recording.check_if_json_serializable():
        zarr_group.attrs["provenance"] = check_json(recording.to_dict(recursive=True))
    else:
        zarr_group.attrs["provenance"] = None

    # save data (done the subclass)
    zarr_group.attrs["sampling_frequency"] = float(recording.get_sampling_frequency())
    zarr_group.attrs["num_segments"] = int(recording.get_num_segments())
    zarr_group.create_dataset(name="channel_ids", data=recording.get_channel_ids(), compressor=None)
    dataset_paths = [f"traces_seg{i}" for i in range(recording.get_num_segments())]

    dtype = recording.get_dtype() if dtype is None else dtype
    channel_chunk_size = zarr_kwargs.get("channel_chunk_size", None)
    global_compressor = zarr_kwargs.pop("compressor", get_default_zarr_compressor())
    compressor_by_dataset = zarr_kwargs.pop("compressor_by_dataset", {})
    global_filters = zarr_kwargs.pop("filters", None)
    filters_by_dataset = zarr_kwargs.pop("filters_by_dataset", {})

    compressor_traces = compressor_by_dataset.get("traces", global_compressor)
    filters_traces = filters_by_dataset.get("traces", global_filters)
    add_traces_to_zarr(
        recording=recording,
        zarr_group=zarr_group,
        dataset_paths=dataset_paths,
        compressor=compressor_traces,
        filters=filters_traces,
        dtype=dtype,
        channel_chunk_size=channel_chunk_size,
        auto_cast_uint=auto_cast_uint,
        verbose=verbose,
        **job_kwargs,
    )

    # save probe
    if recording.get_property("contact_vector") is not None:
        probegroup = recording.get_probegroup()
        zarr_group.attrs["probe"] = check_json(probegroup.to_dict(array_as_list=True))

    # save time vector if any
    t_starts = np.zeros(recording.get_num_segments(), dtype="float64") * np.nan
    for segment_index, rs in enumerate(recording._recording_segments):
        d = rs.get_times_kwargs()
        time_vector = d["time_vector"]

        compressor_times = compressor_by_dataset.get("times", global_compressor)
        filters_times = filters_by_dataset.get("times", global_filters)

        if time_vector is not None:
            _ = zarr_group.create_dataset(
                name=f"times_seg{segment_index}",
                data=time_vector,
                filters=filters_times,
                compressor=compressor_times,
            )
        elif d["t_start"] is not None:
            t_starts[segment_index] = d["t_start"]

    if np.any(~np.isnan(t_starts)):
        zarr_group.create_dataset(name="t_starts", data=t_starts, compressor=None)

    add_properties_and_annotations(zarr_group, recording)


def add_traces_to_zarr(
    recording,
    zarr_group,
    dataset_paths,
    channel_chunk_size=None,
    dtype=None,
    compressor=None,
    filters=None,
    verbose=False,
    auto_cast_uint=True,
    **job_kwargs,
):
    """
    Save the trace of a recording extractor in several zarr format.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object to be saved in .dat format
    zarr_group : zarr.Group
        The zarr group to add traces to
    dataset_paths : list
        List of paths to traces datasets in the zarr group
    channel_chunk_size : int or None, default: None (chunking in time only)
        Channels per chunk
    dtype : dtype, default: None
        Type of the saved data
    compressor : zarr compressor or None, default: None
        Zarr compressor
    filters : list, default: None
        List of zarr filters
    verbose : bool, default: False
        If True, output is verbose (when chunks are used)
    auto_cast_uint : bool, default: True
        If True, unsigned integers are automatically cast to int if the specified dtype is signed
    {}
    """
    from .job_tools import (
        ensure_chunk_size,
        fix_job_kwargs,
        ChunkRecordingExecutor,
    )

    assert dataset_paths is not None, "Provide 'file_path'"

    if not isinstance(dataset_paths, list):
        dataset_paths = [dataset_paths]
    assert len(dataset_paths) == recording.get_num_segments()

    if dtype is None:
        dtype = recording.get_dtype()
    if auto_cast_uint:
        cast_unsigned = determine_cast_unsigned(recording, dtype)
    else:
        cast_unsigned = False

    job_kwargs = fix_job_kwargs(job_kwargs)
    chunk_size = ensure_chunk_size(recording, **job_kwargs)

    # create zarr datasets files
    zarr_datasets = []
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        num_channels = recording.get_num_channels()
        dset_name = dataset_paths[segment_index]
        shape = (num_frames, num_channels)
        dset = zarr_group.create_dataset(
            name=dset_name,
            shape=shape,
            chunks=(chunk_size, channel_chunk_size),
            dtype=dtype,
            filters=filters,
            compressor=compressor,
        )
        zarr_datasets.append(dset)
        # synchronizer=zarr.ThreadSynchronizer())

    # use executor (loop or workers)
    func = _write_zarr_chunk
    init_func = _init_zarr_worker
    init_args = (recording, zarr_datasets, dtype, cast_unsigned)
    executor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, verbose=verbose, job_name="write_zarr_recording", **job_kwargs
    )
    executor.run()


# used by write_zarr_recording + ChunkRecordingExecutor
def _init_zarr_worker(recording, zarr_datasets, dtype, cast_unsigned):
    import zarr

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["zarr_datasets"] = zarr_datasets
    worker_ctx["dtype"] = np.dtype(dtype)
    worker_ctx["cast_unsigned"] = cast_unsigned

    return worker_ctx


# used by write_zarr_recording + ChunkRecordingExecutor
def _write_zarr_chunk(segment_index, start_frame, end_frame, worker_ctx):
    import gc

    # recover variables of the worker
    recording = worker_ctx["recording"]
    dtype = worker_ctx["dtype"]
    zarr_dataset = worker_ctx["zarr_datasets"][segment_index]
    cast_unsigned = worker_ctx["cast_unsigned"]

    # apply function
    traces = recording.get_traces(
        start_frame=start_frame, end_frame=end_frame, segment_index=segment_index, cast_unsigned=cast_unsigned
    )
    traces = traces.astype(dtype)
    zarr_dataset[start_frame:end_frame, :] = traces

    # fix memory leak by forcing garbage collection
    del traces
    gc.collect()
