from __future__ import annotations
from pathlib import Path
import warnings


import numpy as np

from .core_tools import add_suffix, make_shared_array
from .job_tools import (
    chunk_duration_to_chunk_size,
    ensure_n_jobs,
    fix_job_kwargs,
    ChunkExecutor,
    _shared_job_kwargs_doc,
)


def write_binary(
    chunkable: "ChunkableMixin",
    file_paths: list[Path | str] | Path | str,
    dtype: np.typing.DTypeLike = None,
    add_file_extension: bool = True,
    byte_offset: int = 0,
    verbose: bool = False,
    **job_kwargs,
):
    """
    Save the data of a chunkable object to binary format.

    Note :
        time_axis is always 0 (contrary to previous version.
        to get time_axis=1 (which is a bad idea) use `write_binary_file_handle()`

    Parameters
    ----------
    chunkable : ChunkableMixin
        The chunkable object to be saved to binary file
    file_path : str or list[str]
        The path to the file.
    dtype : dtype or None, default: None
        Type of the saved data
    add_file_extension, bool, default: True
        If True, and  the file path does not end in "raw", "bin", or "dat" then "raw" is added as an extension.
    byte_offset : int, default: 0
        Offset in bytes for the binary file (e.g. to write a header). This is useful in case you want to append data
        to an existing file where you wrote a header or other data before.
    verbose : bool
        This is the verbosity of the ChunkExecutor
    {}
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    file_path_list = [file_paths] if not isinstance(file_paths, list) else file_paths
    num_segments = chunkable.get_num_segments()
    if len(file_path_list) != num_segments:
        raise ValueError("'file_paths' must be a list of the same size as the number of segments in the chunkable")

    file_path_list = [Path(file_path) for file_path in file_path_list]
    if add_file_extension:
        file_path_list = [add_suffix(file_path, ["raw", "bin", "dat"]) for file_path in file_path_list]

    dtype = dtype if dtype is not None else chunkable.get_dtype()

    sample_size_bytes = chunkable.get_sample_size_in_bytes()

    file_path_dict = {segment_index: file_path for segment_index, file_path in enumerate(file_path_list)}
    for segment_index, file_path in file_path_dict.items():
        num_samples = chunkable.get_num_samples(segment_index=segment_index)
        data_size_bytes = sample_size_bytes * num_samples
        file_size_bytes = data_size_bytes + byte_offset

        # Create an empty file with file_size_bytes
        with open(file_path, "wb+") as file:
            # The previous implementation `file.truncate(file_size_bytes)` was slow on Windows (#3408)
            file.seek(file_size_bytes - 1)
            file.write(b"\0")

        assert Path(file_path).is_file()

    # use executor (loop or workers)
    func = _write_binary_chunk
    init_func = _init_binary_worker
    init_args = (chunkable, file_path_dict, dtype, byte_offset)
    executor = ChunkExecutor(
        chunkable, func, init_func, init_args, job_name="write_binary", verbose=verbose, **job_kwargs
    )
    executor.run()


# used by write_binary + ChunkExecutor
def _init_binary_worker(chunkable, file_path_dict, dtype, byte_offset):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["chunkable"] = chunkable
    worker_ctx["byte_offset"] = byte_offset
    worker_ctx["dtype"] = np.dtype(dtype)

    file_dict = {segment_index: open(file_path, "rb+") for segment_index, file_path in file_path_dict.items()}
    worker_ctx["file_dict"] = file_dict

    return worker_ctx


# used by write_binary + ChunkExecutor
def _write_binary_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    chunkable = worker_ctx["chunkable"]
    dtype = worker_ctx["dtype"]
    byte_offset = worker_ctx["byte_offset"]
    file = worker_ctx["file_dict"][segment_index]

    sample_size_bytes = chunkable.get_sample_size_in_bytes()

    # Calculate byte offsets for the start frames relative to the entire recording
    start_byte = byte_offset + start_frame * sample_size_bytes

    traces = chunkable.get_data(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    traces = traces.astype(dtype, order="c", copy=False)

    file.seek(start_byte)
    file.write(traces.data)
    # flush is important!!
    file.flush()


write_binary.__doc__ = write_binary.__doc__.format(_shared_job_kwargs_doc)


# used by write_memory_recording
def _init_memory_worker(chunkable, arrays, shm_names, shapes, dtype):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["chunkable"] = chunkable
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


# used by write_memory_recording
def _write_memory_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    chunkable = worker_ctx["chunkable"]
    dtype = worker_ctx["dtype"]
    arr = worker_ctx["arrays"][segment_index]

    # apply function
    traces = chunkable.get_data(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    traces = traces.astype(dtype, copy=False)
    arr[start_frame:end_frame, :] = traces


def write_memory(chunkable, dtype=None, verbose=False, buffer_type="auto", **job_kwargs):
    """
    Save the traces into numpy arrays (memory).
    try to use the SharedMemory introduce in py3.8 if n_jobs > 1

    Parameters
    ----------
    chunkable : ChunkableMixin
        The chunkable object to be saved to memory
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
        dtype = chunkable.get_dtype()

    # create sharedmmep
    arrays = []
    shm_names = []
    shms = []
    shapes = []

    n_jobs = ensure_n_jobs(chunkable, n_jobs=job_kwargs.get("n_jobs", 1))
    if buffer_type == "auto":
        if n_jobs > 1:
            buffer_type = "sharedmem"
        else:
            buffer_type = "numpy"

    for segment_index in range(chunkable.get_num_segments()):
        shape = chunkable.get_shape()
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
        init_args = (chunkable, None, shm_names, shapes, dtype)
    else:
        init_args = (chunkable, arrays, None, None, dtype)

    executor = ChunkExecutor(chunkable, func, init_func, init_args, verbose=verbose, job_name=job_name, **job_kwargs)
    executor.run()

    return arrays, shms


write_memory.__doc__ = write_memory.__doc__.format(_shared_job_kwargs_doc)


def get_random_slices(
    chunkable: "ChunkableMixin",
    method="full_random",
    num_chunks_per_segment=20,
    chunk_duration="500ms",
    chunk_size=None,
    margin_frames=0,
    seed=None,
):
    """
    Get random slice of a chunkable object across segments.

    This is used for instance in get_noise_levels() and get_random_data_chunks() to estimate noise on traces.

    Parameters
    ----------
    chunkable : ChunkableMixin
        The chunkable object to get random chunks from
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
                chunk_size = chunk_duration_to_chunk_size(chunk_duration, chunkable)
            else:
                raise ValueError("get_random_slices need chunk_size or chunk_duration")

        # check chunk size
        num_segments = chunkable.get_num_segments()
        for segment_index in range(num_segments):
            chunk_size_limit = chunkable.get_num_samples(segment_index) - 2 * margin_frames
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
            num_frames = chunkable.get_num_samples(segment_index)
            high = num_frames - chunk_size - margin_frames
            # here we set endpoint to True, because the this represents the start of the
            # chunk, and should be inclusive
            random_starts = rng.integers(low=low, high=high, size=size, endpoint=True)
            random_starts = np.sort(random_starts)
            slices += [(segment_index, start_frame, (start_frame + chunk_size)) for start_frame in random_starts]
    else:
        raise ValueError(f"get_random_slices : wrong method {method}")

    return slices


def get_chunks(chunkable: "ChunkableMixin", concatenated=True, get_data_kwargs=None, **random_slices_kwargs):
    """
    Extract random chunks across segments.

    Internally, it uses `get_random_slices()` and retrieves the traces chunk as a list
    or a concatenated unique array.

    Please read `get_random_slices()` for more details on parameters.

    # TODO: handle this in recording tools:
    return * will be get_data_kwargs

    Parameters
    ----------
    chunkable : ChunkableMixin
        The chunkable object to get random chunks from
    return_scaled : bool | None, default: None
        DEPRECATED. Use return_in_uV instead.
    return_in_uV : bool, default: False
        If True and the chunkable has scaling (gain_to_uV and offset_to_uV properties),
        traces are scaled to uV
    num_chunks_per_segment : int, default: 20
        Number of chunks per segment
    concatenated : bool, default: True
        If True chunk are concatenated along time axis
    **random_slices_kwargs : dict
        Options transmited to  get_random_slices(), please read documentation from this
        function for more details.

    Returns
    -------
    chunk_list : np.array | list of np.array
        Array of concatenate chunks per segment
    """
    # Handle deprecated return_scaled parameter
    if return_scaled is not None:
        warnings.warn(
            "`return_scaled` is deprecated and will be removed in version 0.105.0. Use `return_in_uV` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return_in_uV = return_scaled

    slices = get_random_slices(chunkable, **random_slices_kwargs)

    chunk_list = []
    for segment_index, start_frame, end_frame in slices:
        traces_chunk = chunkable.get_data(
            start_frame=start_frame, end_frame=end_frame, segment_index=segment_index, **get_data_kwargs
        )
        chunk_list.append(traces_chunk)

    if concatenated:
        return np.concatenate(chunk_list, axis=0)
    else:
        return chunk_list
