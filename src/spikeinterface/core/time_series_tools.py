from pathlib import Path
import warnings


import numpy as np

from .core_tools import add_suffix, make_shared_array
from .job_tools import (
    chunk_duration_to_chunk_size,
    ensure_n_jobs,
    fix_job_kwargs,
    TimeSeriesChunkExecutor,
    _shared_job_kwargs_doc,
)
from .time_series import TimeSeries, TimeSeriesSegment


def write_binary(
    time_series: TimeSeries,
    file_paths: list[Path | str] | Path | str,
    file_timestamps_paths: list[Path | str] | Path | str | None = None,
    dtype: np.typing.DTypeLike = None,
    add_file_extension: bool = True,
    byte_offset: int = 0,
    verbose: bool = False,
    **job_kwargs,
):
    """
    Save the data of a time_series object to binary format.

    Note :
        time_axis is always 0 (contrary to previous version.
        to get time_axis=1 (which is a bad idea) use `write_binary_file_handle()`

    Parameters
    ----------
    time_series : TimeSeries
        The time_series object to be saved to binary file
    file_paths : list[Path | str] | Path | str
        The path to the files to save data for each segment.
    file_timestamps_paths : list[Path | str] | Path | str | None, default: None
        The path to the timestamps file. If None, timestamps are not saved.
    dtype : dtype or None, default: None
        Type of the saved data
    add_file_extension, bool, default: True
        If True, and  the file path does not end in "raw", "bin", or "dat" then "raw" is added as an extension.
    byte_offset : int, default: 0
        Offset in bytes for the binary file (e.g. to write a header). This is useful in case you want to append data
        to an existing file where you wrote a header or other data before.
    verbose : bool
        This is the verbosity of the TimeSeriesChunkExecutor
    {}
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    file_path_list = [file_paths] if not isinstance(file_paths, list) else file_paths
    num_segments = time_series.get_num_segments()
    if len(file_path_list) != num_segments:
        raise ValueError("'file_paths' must be a list of the same size as the number of segments in the time_series")

    file_path_list = [Path(file_path) for file_path in file_path_list]
    if add_file_extension:
        file_path_list = [add_suffix(file_path, ["raw", "bin", "dat"]) for file_path in file_path_list]

    dtype = dtype if dtype is not None else time_series.get_dtype()

    sample_size_bytes = time_series.get_sample_size_in_bytes()

    file_path_dict = {segment_index: file_path for segment_index, file_path in enumerate(file_path_list)}
    if file_timestamps_paths is not None:
        file_timestamps_path_dict = {
            segment_index: file_path for segment_index, file_path in enumerate(file_timestamps_paths)
        }
    else:
        file_timestamps_path_dict = None
    for segment_index, file_path in file_path_dict.items():
        num_samples = time_series.get_num_samples(segment_index=segment_index)
        data_size_bytes = sample_size_bytes * num_samples
        file_size_bytes = data_size_bytes + byte_offset

        # Create an empty file with file_size_bytes
        with open(file_path, "wb+") as file:
            # The previous implementation `file.truncate(file_size_bytes)` was slow on Windows (#3408)
            file.seek(file_size_bytes - 1)
            file.write(b"\0")

        if file_timestamps_path_dict is not None:
            file_timestamps_path = file_timestamps_path_dict[segment_index]
            with open(file_timestamps_path, "wb+") as file:
                file.seek(num_samples * 8 - 1)  # 8 bytes for float64 timestamps
                file.write(b"\0")

        assert Path(file_path).is_file()

    # use executor (loop or workers)
    func = _write_binary_chunk
    init_func = _init_binary_worker
    init_args = (time_series, file_path_dict, dtype, byte_offset, file_timestamps_path_dict)
    executor = TimeSeriesChunkExecutor(
        time_series, func, init_func, init_args, job_name="write_binary", verbose=verbose, **job_kwargs
    )
    executor.run()


# used by write_binary + TimeSeriesChunkExecutor
def _init_binary_worker(time_series, file_path_dict, dtype, byte_offset, file_timestamps_path_dict=None):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["time_series"] = time_series
    worker_ctx["byte_offset"] = byte_offset
    worker_ctx["dtype"] = np.dtype(dtype)

    file_dict = {segment_index: open(file_path, "rb+") for segment_index, file_path in file_path_dict.items()}
    worker_ctx["file_dict"] = file_dict
    worker_ctx["file_timestamps_dict"] = file_timestamps_path_dict

    return worker_ctx


# used by write_binary + TimeSeriesChunkExecutor
def _write_binary_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    time_series = worker_ctx["time_series"]
    dtype = worker_ctx["dtype"]
    byte_offset = worker_ctx["byte_offset"]
    file = worker_ctx["file_dict"][segment_index]
    file_timestamps_dict = worker_ctx["file_timestamps_dict"]
    sample_size_bytes = time_series.get_sample_size_in_bytes()

    # Calculate byte offsets for the start frames relative to the entire recording
    start_byte = byte_offset + start_frame * sample_size_bytes

    data = time_series.get_data(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    data = data.astype(dtype, order="c", copy=False)

    file.seek(start_byte)
    file.write(data.data)
    # flush is important!!
    file.flush()

    if file_timestamps_dict is not None:
        file_timestamps = file_timestamps_dict[segment_index]
        timestamps = time_series.get_times(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
        timestamps = timestamps.astype("float64", order="c", copy=False)
        timestamp_byte_offset = start_frame * 8  # 8 bytes for float64
        file.seek(timestamp_byte_offset)
        file.write(timestamps.data)
        file.flush()


write_binary.__doc__ = write_binary.__doc__.format(_shared_job_kwargs_doc)


# used by write_memory
def _init_memory_worker(time_series, arrays, shm_names, shapes, dtype):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["time_series"] = time_series
    worker_ctx["dtype"] = np.dtype(dtype)

    if arrays is None:
        # create it from share memory name
        from multiprocessing.shared_memory import SharedMemory

        arrays = []
        # keep shm alive
        worker_ctx["shms"] = []
        for i in range(len(shm_names)):
            shm = SharedMemory(shm_names[i])
            worker_ctx["shms"].append(shm)
            arr = np.ndarray(shape=shapes[i], dtype=dtype, buffer=shm.buf)
            arrays.append(arr)

    worker_ctx["arrays"] = arrays

    return worker_ctx


# used by write_memory
def _write_memory_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    time_series = worker_ctx["time_series"]
    dtype = worker_ctx["dtype"]
    arr = worker_ctx["arrays"][segment_index]

    # apply function
    traces = time_series.get_data(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    traces = traces.astype(dtype, copy=False)
    arr[start_frame:end_frame, :] = traces


def write_memory(
    time_series: TimeSeries, dtype=None, verbose=False, buffer_type="auto", job_name="write_memory", **job_kwargs
):
    """
    Save the traces into numpy arrays (memory).
    try to use the SharedMemory introduce in py3.8 if n_jobs > 1

    Parameters
    ----------
    time_series : TimeSeries
        The time_series object to be saved to memory
    dtype : dtype, default: None
        Type of the saved data
    verbose : bool, default: False
        If True, output is verbose (when chunks are used)
    buffer_type : "auto" | "numpy" | "sharedmem",
        The type of buffer to use for storing the data.
    job_name : str, default: "write_memory"
        Name of the job
    {}

    Returns
    ---------
    arrays : one array per segment
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    if dtype is None:
        dtype = time_series.get_dtype()

    # create sharedmmep
    arrays = []
    shm_names = []
    shms = []
    shapes = []

    n_jobs = ensure_n_jobs(time_series, n_jobs=job_kwargs.get("n_jobs", 1))
    if buffer_type == "auto":
        if n_jobs > 1:
            buffer_type = "sharedmem"
        else:
            buffer_type = "numpy"

    for segment_index in range(time_series.get_num_segments()):
        shape = time_series.get_shape(segment_index=segment_index)
        shapes.append(shape)
        if buffer_type == "sharedmem":
            arr, shm = make_shared_array(shape, dtype)
            shm_names.append(shm.name)
            shms.append(shm)
        else:
            arr = np.zeros(shape, dtype=dtype)
            shms.append(None)
        arrays.append(arr)

    # use executor (loop or workers)
    func = _write_memory_chunk
    init_func = _init_memory_worker
    if n_jobs > 1:
        init_args = (time_series, None, shm_names, shapes, dtype)
    else:
        init_args = (time_series, arrays, None, None, dtype)

    executor = TimeSeriesChunkExecutor(
        time_series, func, init_func, init_args, verbose=verbose, job_name=job_name, **job_kwargs
    )
    executor.run()

    return arrays, shms


write_memory.__doc__ = write_memory.__doc__.format(_shared_job_kwargs_doc)


def _write_time_series_to_zarr(
    time_series: TimeSeries,
    zarr_group,
    dataset_paths,
    dataset_timestamps_paths=None,
    extra_chunks=None,
    dtype=None,
    compressor_data=None,
    filters_data=None,
    compressor_times=None,
    filters_times=None,
    verbose=False,
    **job_kwargs,
):
    """
    Save the trace of a time_series object in several zarr format.

    Parameters
    ----------
    time_series : TimeSeries
        The time_series object to be saved in .dat format
    zarr_group : zarr.Group
        The zarr group to add traces to
    dataset_paths : list
        List of paths to traces datasets in the zarr group
    dataset_timestamps_paths : list or None, default: None
        List of paths to timestamps datasets in the zarr group. If None, timestamps are not saved.
    extra_chunks : tuple or None, default: None
        Extra chunking dimensions to use for the zarr dataset.
        The first dimension is always time and controlled by the job_kwargs.
        This is for example useful to chunk by channel, with `extra_chunks=(channel_chunk_size,)`.
    dtype : dtype, default: None
        Type of the saved data
    compressor_data : zarr compressor or None, default: None
        Zarr compressor for data
    filters_data : list, default: None
        List of zarr filters for data
    compressor_times : zarr compressor or None, default: None
        Zarr compressor for timestamps
    filters_times : list, default: None
        List of zarr filters for timestamps
    verbose : bool, default: False
        If True, output is verbose (when chunks are used)
    {}
    """
    from .job_tools import (
        ensure_chunk_size,
        fix_job_kwargs,
        TimeSeriesChunkExecutor,
    )

    assert dataset_paths is not None, "Provide 'dataset_paths' to save data in zarr format"
    if dataset_timestamps_paths is not None:
        assert (
            len(dataset_timestamps_paths) == time_series.get_num_segments()
        ), "dataset_timestamps_paths should have the same length as the number of segments in the time_series"
    else:
        dataset_timestamps_paths = [None] * time_series.get_num_segments()

    if not isinstance(dataset_paths, list):
        dataset_paths = [dataset_paths]
    assert len(dataset_paths) == time_series.get_num_segments()

    if dtype is None:
        dtype = time_series.get_dtype()

    job_kwargs = fix_job_kwargs(job_kwargs)
    chunk_size = ensure_chunk_size(time_series, **job_kwargs)

    if extra_chunks is not None:
        assert len(extra_chunks) == len(time_series.get_shape(0)[1:]), (
            "extra_chunks should have the same length as the number of dimensions "
            "of the time_series minus one (time axis)"
        )

    # create zarr datasets files
    zarr_datasets = []
    zarr_timestamps_datasets = []

    for segment_index in range(time_series.get_num_segments()):
        num_samples = time_series.get_num_samples(segment_index)
        dset_name = dataset_paths[segment_index]
        shape = time_series.get_shape(segment_index)
        dset = zarr_group.create_dataset(
            name=dset_name,
            shape=shape,
            chunks=(chunk_size,) + extra_chunks if extra_chunks is not None else (chunk_size,),
            dtype=dtype,
            filters=filters_data,
            compressor=compressor_data,
        )
        zarr_datasets.append(dset)
        if dataset_timestamps_paths[segment_index] is not None:
            tset_name = dataset_timestamps_paths[segment_index]
            zarr_timestamps_datasets.append(
                zarr_group.create_dataset(
                    name=tset_name,
                    shape=(num_samples,),
                    chunks=(chunk_size,),
                    dtype="float64",
                    filters=filters_times,
                    compressor=compressor_times,
                )
            )
        else:
            zarr_timestamps_datasets.append(None)

    # use executor (loop or workers)
    func = _write_zarr_chunk
    init_func = _init_zarr_worker
    init_args = (time_series, zarr_datasets, dtype, zarr_timestamps_datasets)
    executor = TimeSeriesChunkExecutor(
        time_series, func, init_func, init_args, verbose=verbose, job_name="write_zarr", **job_kwargs
    )
    executor.run()

    # save t_starts
    t_starts = np.zeros(time_series.get_num_segments(), dtype="float64") * np.nan
    for segment_index in range(time_series.get_num_segments()):
        time_info = time_series.get_time_info(segment_index)
        if time_info["t_start"] is not None:
            t_starts[segment_index] = time_info["t_start"]

    if np.any(~np.isnan(t_starts)):
        zarr_group.create_dataset(name="t_starts", data=t_starts, compressor=None)


def _init_zarr_worker(time_series, zarr_datasets, dtype, zarr_timestamps_datasets=None):
    import zarr

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["time_series"] = time_series
    worker_ctx["zarr_datasets"] = zarr_datasets
    if zarr_timestamps_datasets is not None and len(zarr_timestamps_datasets) > 0:
        worker_ctx["zarr_timestamps_datasets"] = zarr_timestamps_datasets
    else:
        worker_ctx["zarr_timestamps_datasets"] = None
    worker_ctx["dtype"] = np.dtype(dtype)

    return worker_ctx


def _write_zarr_chunk(segment_index, start_frame, end_frame, worker_ctx):
    import gc

    # recover variables of the worker
    time_series = worker_ctx["time_series"]
    dtype = worker_ctx["dtype"]
    zarr_dataset = worker_ctx["zarr_datasets"][segment_index]
    if worker_ctx["zarr_timestamps_datasets"] is not None:
        zarr_timestamps_dataset = worker_ctx["zarr_timestamps_datasets"][segment_index]
    else:
        zarr_timestamps_dataset = None

    # apply function
    data = time_series.get_data(
        start_frame=start_frame,
        end_frame=end_frame,
        segment_index=segment_index,
    )
    data = data.astype(dtype)
    zarr_dataset[start_frame:end_frame, :] = data

    if zarr_timestamps_dataset is not None:
        timestamps = time_series.get_times(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
        zarr_timestamps_dataset[start_frame:end_frame] = timestamps

    # fix memory leak by forcing garbage collection
    del data
    gc.collect()


def get_random_sample_slices(
    time_series: TimeSeries,
    method="full_random",
    num_chunks_per_segment=20,
    chunk_duration="500ms",
    chunk_size=None,
    margin_frames=0,
    seed=None,
):
    """
    Get random slice of a time_series object across segments.

    Parameters
    ----------
    time_series : TimeSeries
        The time_series object to get random chunks from
    method : "full_random"
        The method used to get random slices.
          * "full_random" : legacy method,  used until version 0.101.0, there is no constrain on slices
            and they can overlap.
    num_chunks_per_segment : int, default: 20
        Number of chunks per segment
    chunk_duration : str | float | None, default "500ms"
        The duration of each chunk in 's' or 'ms'
    chunk_size : int | None
        Size of a chunk in number of frames. This is used only if chunk_duration is None.
        This is kept for backward compatibility, you should prefer 'chunk_duration=500ms' instead.
    concatenated : bool, default: True
        If True chunk are concatenated along time axis
    seed : int, default: None
        Random seed
    margin_frames : int, default: 0
        Margin in number of frames to avoid edge effects

    Returns
    -------
    chunk_list : np.array
        Array of concatenate chunks per segment


    """
    # TODO: if segment have differents length make another sampling that dependant on the length of the segment
    # Should be done by changing kwargs with total_num_chunks=XXX and total_duration=YYYY
    # And randomize the number of chunk per segment weighted by segment duration

    if method == "full_random":
        if chunk_size is None:
            if chunk_duration is not None:
                chunk_size = chunk_duration_to_chunk_size(chunk_duration, time_series)
            else:
                raise ValueError("get_random_sample_slices need chunk_size or chunk_duration")

        # check chunk size
        num_segments = time_series.get_num_segments()
        for segment_index in range(num_segments):
            chunk_size_limit = time_series.get_num_samples(segment_index) - 2 * margin_frames
            if chunk_size > chunk_size_limit:
                chunk_size = chunk_size_limit - 1
                warnings.warn(
                    f"chunk_size is greater than the number "
                    f"of samples for segment index {segment_index}. "
                    f"Using {chunk_size}."
                )
        rng = np.random.default_rng(seed)
        slices = []
        low = margin_frames
        size = num_chunks_per_segment
        for segment_index in range(num_segments):
            num_frames = time_series.get_num_samples(segment_index)
            high = num_frames - chunk_size - margin_frames
            # here we set endpoint to True, because the this represents the start of the
            # chunk, and should be inclusive
            random_starts = rng.integers(low=low, high=high, size=size, endpoint=True)
            random_starts = np.sort(random_starts)
            slices += [(segment_index, start_frame, (start_frame + chunk_size)) for start_frame in random_starts]
    else:
        raise ValueError(f"get_random_sample_slices : wrong method {method}")

    return slices


def get_chunks(time_series: TimeSeries, concatenated=True, get_data_kwargs=None, **random_slices_kwargs):
    """
    Extract random chunks across segments.

    Internally, it uses `get_random_sample_slices()` and retrieves the traces chunk as a list
    or a concatenated unique array.

    Please read `get_random_sample_slices()` for more details on parameters.

    # TODO: handle this in recording tools:
    return * will be get_data_kwargs

    Parameters
    ----------
    time_series : TimeSeries
        The time_series object to get random chunks from
    return_scaled : bool | None, default: None
        DEPRECATED. Use return_in_uV instead.
    return_in_uV : bool, default: False
        If True and the time_series has scaling (gain_to_uV and offset_to_uV properties),
        traces are scaled to uV
    num_chunks_per_segment : int, default: 20
        Number of chunks per segment
    concatenated : bool, default: True
        If True chunk are concatenated along time axis
    **random_slices_kwargs : dict
        Options transmited to  get_random_sample_slices(), please read documentation from this
        function for more details.

    Returns
    -------
    chunk_list : np.ndarray | list of np.array
        Array of concatenate chunks per segment
    """
    slices = get_random_sample_slices(time_series, **random_slices_kwargs)

    chunk_list = []
    get_data_kwargs = get_data_kwargs if get_data_kwargs is not None else {}
    for segment_index, start_frame, end_frame in slices:
        traces_chunk = time_series.get_data(
            start_frame=start_frame, end_frame=end_frame, segment_index=segment_index, **get_data_kwargs
        )
        chunk_list.append(traces_chunk)

    if concatenated:
        return np.concatenate(chunk_list, axis=0)
    else:
        return chunk_list


def get_time_series_chunk_with_margin(
    chunkable_segment: TimeSeriesSegment,
    start_frame,
    end_frame,
    last_dimension_indices,
    margin,
    add_zeros=False,
    add_reflect_padding=False,
    window_on_margin=False,
    dtype=None,
):
    """
    Helper to get chunk with margin

    The margin is extracted from the recording when possible. If
    at the edge of the recording, no margin is used unless one
    of `add_zeros` or `add_reflect_padding` is True. In the first
    case zero padding is used, in the second case np.pad is called
    with mod="reflect".
    """
    length = int(chunkable_segment.get_num_samples())

    if last_dimension_indices is None:
        last_dimension_indices = slice(None)

    if not (add_zeros or add_reflect_padding):
        if window_on_margin and not add_zeros:
            raise ValueError("window_on_margin requires add_zeros=True")

        if start_frame is None:
            left_margin = 0
            start_frame = 0
        elif start_frame < margin:
            left_margin = start_frame
        else:
            left_margin = margin

        if end_frame is None:
            right_margin = 0
            end_frame = length
        elif end_frame > (length - margin):
            right_margin = length - end_frame
        else:
            right_margin = margin

        data_chunk = chunkable_segment.get_data(
            start_frame - left_margin,
            end_frame + right_margin,
            last_dimension_indices,
        )

    else:
        # either add_zeros or reflect_padding
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = length

        chunk_size = end_frame - start_frame
        full_size = chunk_size + 2 * margin

        if start_frame < margin:
            start_frame2 = 0
            left_pad = margin - start_frame
        else:
            start_frame2 = start_frame - margin
            left_pad = 0

        if end_frame > (length - margin):
            end_frame2 = length
            right_pad = end_frame + margin - length
        else:
            end_frame2 = end_frame + margin
            right_pad = 0

        data_chunk = chunkable_segment.get_data(start_frame2, end_frame2, last_dimension_indices)

        if dtype is not None or window_on_margin or left_pad > 0 or right_pad > 0:
            need_copy = True
        else:
            need_copy = False

        left_margin = margin
        right_margin = margin

        if need_copy:
            if dtype is None:
                dtype = data_chunk.dtype

            left_margin = margin
            if end_frame < (length + margin):
                right_margin = margin
            else:
                right_margin = end_frame + margin - length

            if add_zeros:
                data_chunk2 = np.zeros((full_size, data_chunk.shape[1]), dtype=dtype)
                i0 = left_pad
                i1 = left_pad + data_chunk.shape[0]
                data_chunk2[i0:i1, :] = data_chunk
                if window_on_margin:
                    # apply inplace taper on border
                    taper = (1 - np.cos(np.arange(margin) / margin * np.pi)) / 2
                    taper = taper[:, np.newaxis]
                    data_chunk2[:margin] *= taper
                    data_chunk2[-margin:] *= taper[::-1]
                data_chunk = data_chunk2
            elif add_reflect_padding:
                # in this case, we don't want to taper
                data_chunk = np.pad(
                    data_chunk.astype(dtype, copy=False),
                    [(left_pad, right_pad)] + [(0, 0)] * (data_chunk.ndim - 1),
                    mode="reflect",
                )
            else:
                # we need a copy to change the dtype
                data_chunk = np.asarray(data_chunk, dtype=dtype)

    return data_chunk, left_margin, right_margin
