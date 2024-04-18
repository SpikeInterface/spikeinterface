from __future__ import annotations
from copy import deepcopy
from typing import Literal
import warnings
from pathlib import Path
import os
import gc
import mmap
import tqdm


import numpy as np

from .core_tools import add_suffix, make_shared_array
from .job_tools import (
    ensure_chunk_size,
    ensure_n_jobs,
    divide_segment_into_chunks,
    fix_job_kwargs,
    ChunkRecordingExecutor,
    _shared_job_kwargs_doc,
)


def read_binary_recording(file, num_channels, dtype, time_axis=0, offset=0):
    """
    Read binary .bin or .dat file.

    Parameters
    ----------
    file: str
        File name
    num_channels: int
        Number of channels
    dtype: dtype
        dtype of the file
    time_axis: 0 or 1, default: 0
        If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
        If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
    offset: int, default: 0
        number of offset bytes

    """
    # TODO change this function to read_binary_traces() because this name is confusing
    num_channels = int(num_channels)
    with Path(file).open() as f:
        nsamples = (os.fstat(f.fileno()).st_size - offset) // (num_channels * np.dtype(dtype).itemsize)
    if time_axis == 0:
        samples = np.memmap(file, np.dtype(dtype), mode="r", offset=offset, shape=(nsamples, num_channels))
    else:
        samples = np.memmap(file, np.dtype(dtype), mode="r", offset=offset, shape=(num_channels, nsamples)).T
    return samples


# used by write_binary_recording + ChunkRecordingExecutor
def _init_binary_worker(recording, file_path_dict, dtype, byte_offest, cast_unsigned):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["byte_offset"] = byte_offest
    worker_ctx["dtype"] = np.dtype(dtype)
    worker_ctx["cast_unsigned"] = cast_unsigned

    file_dict = {segment_index: open(file_path, "r+") for segment_index, file_path in file_path_dict.items()}
    worker_ctx["file_dict"] = file_dict

    return worker_ctx


def write_binary_recording(
    recording,
    file_paths,
    dtype=None,
    add_file_extension=True,
    byte_offset=0,
    auto_cast_uint=True,
    **job_kwargs,
):
    """
    Save the trace of a recording extractor in several binary .dat format.

    Note :
        time_axis is always 0 (contrary to previous version.
        to get time_axis=1 (which is a bad idea) use `write_binary_recording_file_handle()`

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    file_path: str or list[str]
        The path to the file.
    dtype: dtype or None, default: None
        Type of the saved data
        If True, file the ".raw" file extension is added if the file name is not a "raw", "bin", or "dat"
    byte_offset: int, default: 0
        Offset in bytes for the binary file (e.g. to write a header)
    auto_cast_uint: bool, default: True
        If True, unsigned integers are automatically cast to int if the specified dtype is signed
    {}
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    file_path_list = [file_paths] if not isinstance(file_paths, list) else file_paths
    num_segments = recording.get_num_segments()
    if len(file_path_list) != num_segments:
        raise ValueError("'file_paths' must be a list of the same size as the number of segments in the recording")

    file_path_list = [Path(file_path) for file_path in file_path_list]
    if add_file_extension:
        file_path_list = [add_suffix(file_path, ["raw", "bin", "dat"]) for file_path in file_path_list]

    dtype = dtype if dtype is not None else recording.get_dtype()
    cast_unsigned = False
    if auto_cast_uint:
        cast_unsigned = determine_cast_unsigned(recording, dtype)

    dtype_size_bytes = np.dtype(dtype).itemsize
    num_channels = recording.get_num_channels()

    file_path_dict = {segment_index: file_path for segment_index, file_path in enumerate(file_path_list)}
    for segment_index, file_path in file_path_dict.items():
        num_frames = recording.get_num_frames(segment_index=segment_index)
        data_size_bytes = dtype_size_bytes * num_frames * num_channels
        file_size_bytes = data_size_bytes + byte_offset

        file = open(file_path, "wb+")
        file.truncate(file_size_bytes)
        file.close()
        assert Path(file_path).is_file()

    # use executor (loop or workers)
    func = _write_binary_chunk
    init_func = _init_binary_worker
    init_args = (recording, file_path_dict, dtype, byte_offset, cast_unsigned)
    executor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, job_name="write_binary_recording", **job_kwargs
    )
    executor.run()


# used by write_binary_recording + ChunkRecordingExecutor
def _write_binary_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx["recording"]
    dtype = worker_ctx["dtype"]
    byte_offset = worker_ctx["byte_offset"]
    cast_unsigned = worker_ctx["cast_unsigned"]
    file = worker_ctx["file_dict"][segment_index]

    # Open the memmap
    # What we need is the file_path
    num_channels = recording.get_num_channels()
    num_frames = recording.get_num_frames(segment_index=segment_index)
    shape = (num_frames, num_channels)
    dtype_size_bytes = np.dtype(dtype).itemsize
    data_size_bytes = dtype_size_bytes * num_frames * num_channels

    # Offset (The offset needs to be multiple of the page size)
    # The mmap offset is associated to be as big as possible but still a multiple of the page size
    # The array offset takes care of the reminder
    mmap_offset, array_offset = divmod(byte_offset, mmap.ALLOCATIONGRANULARITY)
    mmmap_length = data_size_bytes + array_offset
    memmap_obj = mmap.mmap(file.fileno(), length=mmmap_length, access=mmap.ACCESS_WRITE, offset=mmap_offset)

    array = np.ndarray.__new__(np.ndarray, shape=shape, dtype=dtype, buffer=memmap_obj, order="C", offset=array_offset)
    # apply function
    traces = recording.get_traces(
        start_frame=start_frame, end_frame=end_frame, segment_index=segment_index, cast_unsigned=cast_unsigned
    )
    if traces.dtype != dtype:
        traces = traces.astype(dtype, copy=False)
    array[start_frame:end_frame, :] = traces

    # Close the memmap
    memmap_obj.flush()


write_binary_recording.__doc__ = write_binary_recording.__doc__.format(_shared_job_kwargs_doc)


def write_binary_recording_file_handle(
    recording, file_handle=None, time_axis=0, dtype=None, byte_offset=0, verbose=False, **job_kwargs
):
    """
    Old variant version of write_binary_recording with one file handle.
    Can be useful in some case ???
    Not used anymore at the moment.

    @ SAM useful for writing with time_axis=1!
    """
    assert file_handle is not None
    assert recording.get_num_segments() == 1, "If file_handle is given then only deals with one segment"

    if dtype is None:
        dtype = recording.get_dtype()

    job_kwargs = fix_job_kwargs(job_kwargs)
    chunk_size = ensure_chunk_size(recording, **job_kwargs)

    if chunk_size is not None and time_axis == 1:
        print("Chunking disabled due to 'time_axis' == 1")
        chunk_size = None

    if chunk_size is None:
        # no chunking
        traces = recording.get_traces(segment_index=0)
        if time_axis == 1:
            traces = traces.T
        if dtype is not None:
            traces = traces.astype(dtype, copy=False)
        traces.tofile(file_handle)
    else:
        num_frames = recording.get_num_samples(segment_index=0)
        chunks = divide_segment_into_chunks(num_frames, chunk_size)

        for start_frame, end_frame in chunks:
            traces = recording.get_traces(segment_index=0, start_frame=start_frame, end_frame=end_frame)
            if time_axis == 1:
                traces = traces.T
            if dtype is not None:
                traces = traces.astype(dtype, copy=False)
            file_handle.write(traces.tobytes())


# used by write_memory_recording
def _init_memory_worker(recording, arrays, shm_names, shapes, dtype, cast_unsigned):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor

        worker_ctx["recording"] = load_extractor(recording)
    else:
        worker_ctx["recording"] = recording

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
    worker_ctx["cast_unsigned"] = cast_unsigned

    return worker_ctx


# used by write_memory_recording
def _write_memory_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx["recording"]
    dtype = worker_ctx["dtype"]
    arr = worker_ctx["arrays"][segment_index]
    cast_unsigned = worker_ctx["cast_unsigned"]

    # apply function
    traces = recording.get_traces(
        start_frame=start_frame, end_frame=end_frame, segment_index=segment_index, cast_unsigned=cast_unsigned
    )
    traces = traces.astype(dtype, copy=False)
    arr[start_frame:end_frame, :] = traces


def write_memory_recording(recording, dtype=None, verbose=False, auto_cast_uint=True, buffer_type="auto", **job_kwargs):
    """
    Save the traces into numpy arrays (memory).
    try to use the SharedMemory introduce in py3.8 if n_jobs > 1

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    dtype: dtype, default: None
        Type of the saved data
    verbose: bool, default: False
        If True, output is verbose (when chunks are used)
    auto_cast_uint: bool, default: True
        If True, unsigned integers are automatically cast to int if the specified dtype is signed
    buffer_type: "auto" | "numpy" | "sharedmem"
    {}

    Returns
    ---------
    arrays: one array per segment
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    if dtype is None:
        dtype = recording.get_dtype()
    if auto_cast_uint:
        cast_unsigned = determine_cast_unsigned(recording, dtype)
    else:
        cast_unsigned = False

    # create sharedmmep
    arrays = []
    shm_names = []
    shms = []
    shapes = []

    n_jobs = ensure_n_jobs(recording, n_jobs=job_kwargs.get("n_jobs", 1))
    if buffer_type == "auto":
        if n_jobs > 1:
            buffer_type = "sharedmem"
        else:
            buffer_type = "numpy"

    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        num_channels = recording.get_num_channels()
        shape = (num_frames, num_channels)
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
        init_args = (recording, None, shm_names, shapes, dtype, cast_unsigned)
    else:
        init_args = (recording, arrays, None, None, dtype, cast_unsigned)

    if "verbose" in job_kwargs:
        del job_kwargs["verbose"]

    executor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, verbose=verbose, job_name="write_memory_recording", **job_kwargs
    )
    executor.run()

    return arrays, shms


write_memory_recording.__doc__ = write_memory_recording.__doc__.format(_shared_job_kwargs_doc)


def write_to_h5_dataset_format(
    recording,
    dataset_path,
    segment_index,
    save_path=None,
    file_handle=None,
    time_axis=0,
    single_axis=False,
    dtype=None,
    chunk_size=None,
    chunk_memory="500M",
    verbose=False,
    auto_cast_uint=True,
    return_scaled=False,
):
    """
    Save the traces of a recording extractor in an h5 dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    dataset_path: str
        Path to dataset in the h5 file (e.g. "/dataset")
    segment_index: int
        index of segment
    save_path: str, default: None
        The path to the file.
    file_handle: file handle, default: None
        The file handle to dump data. This can be used to append data to an header. In case file_handle is given,
        the file is NOT closed after writing the binary data.
    time_axis: 0 or 1, default: 0
        If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
        If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
    single_axis: bool, default: False
        If True, a single-channel recording is saved as a one dimensional array
    dtype: dtype, default: None
        Type of the saved data
    chunk_size: None or int, default: None
        Number of chunks to save the file in. This avoids too much memory consumption for big files.
        If None and "chunk_memory" is given, the file is saved in chunks of "chunk_memory" MB
    chunk_memory: None or str, default: "500M"
        Chunk size in bytes must end with "k", "M" or "G"
    verbose: bool, default: False
        If True, output is verbose (when chunks are used)
    auto_cast_uint: bool, default: True
        If True, unsigned integers are automatically cast to int if the specified dtype is signed
    return_scaled : bool, default: False
        If True and the recording has scaling (gain_to_uV and offset_to_uV properties),
        traces are dumped to uV
    """
    import h5py

    # ~ assert HAVE_H5, "To write to h5 you need to install h5py: pip install h5py"
    assert save_path is not None or file_handle is not None, "Provide 'save_path' or 'file handle'"

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == "":
            # when suffix is already raw/bin/dat do not change it.
            save_path = save_path.parent / (save_path.name + ".h5")

    num_channels = recording.get_num_channels()
    num_frames = recording.get_num_frames(segment_index=0)

    if file_handle is not None:
        assert isinstance(file_handle, h5py.File)
    else:
        file_handle = h5py.File(save_path, "w")

    if dtype is None:
        dtype_file = recording.get_dtype()
    else:
        dtype_file = dtype
    if auto_cast_uint:
        cast_unsigned = determine_cast_unsigned(recording, dtype)
    else:
        cast_unsigned = False

    if single_axis:
        shape = (num_frames,)
    else:
        if time_axis == 0:
            shape = (num_frames, num_channels)
        else:
            shape = (num_channels, num_frames)

    dset = file_handle.create_dataset(dataset_path, shape=shape, dtype=dtype_file)

    chunk_size = ensure_chunk_size(recording, chunk_size=chunk_size, chunk_memory=chunk_memory, n_jobs=1)

    if chunk_size is None:
        traces = recording.get_traces(cast_unsigned=cast_unsigned, return_scaled=return_scaled)
        if dtype is not None:
            traces = traces.astype(dtype_file, copy=False)
        if time_axis == 1:
            traces = traces.T
        if single_axis:
            dset[:] = traces[:, 0]
        else:
            dset[:] = traces
    else:
        chunk_start = 0
        # chunk size is not None
        n_chunk = num_frames // chunk_size
        if num_frames % chunk_size > 0:
            n_chunk += 1
        if verbose:
            chunks = tqdm(range(n_chunk), ascii=True, desc="Writing to .h5 file")
        else:
            chunks = range(n_chunk)
        for i in chunks:
            traces = recording.get_traces(
                segment_index=segment_index,
                start_frame=i * chunk_size,
                end_frame=min((i + 1) * chunk_size, num_frames),
                cast_unsigned=cast_unsigned,
                return_scaled=return_scaled,
            )
            chunk_frames = traces.shape[0]
            if dtype is not None:
                traces = traces.astype(dtype_file, copy=False)
            if single_axis:
                dset[chunk_start : chunk_start + chunk_frames] = traces[:, 0]
            else:
                if time_axis == 0:
                    dset[chunk_start : chunk_start + chunk_frames, :] = traces
                else:
                    dset[:, chunk_start : chunk_start + chunk_frames] = traces.T

            chunk_start += chunk_frames

    if save_path is not None:
        file_handle.close()
    return save_path


def determine_cast_unsigned(recording, dtype):
    recording_dtype = np.dtype(recording.get_dtype())

    if np.dtype(dtype) != recording_dtype and recording_dtype.kind == "u" and np.dtype(dtype).kind == "i":
        cast_unsigned = True
    else:
        cast_unsigned = False
    return cast_unsigned


def get_random_data_chunks(
    recording,
    return_scaled=False,
    num_chunks_per_segment=20,
    chunk_size=10000,
    concatenated=True,
    seed=0,
    margin_frames=0,
):
    """
    Extract random chunks across segments

    This is used for instance in get_noise_levels() to estimate noise on traces.

    Parameters
    ----------
    recording: BaseRecording
        The recording to get random chunks from
    return_scaled: bool, default: False
        If True, returned chunks are scaled to uV
    num_chunks_per_segment: int, default: 20
        Number of chunks per segment
    chunk_size: int, default: 10000
        Size of a chunk in number of frames
    concatenated: bool, default: True
        If True chunk are concatenated along time axis
    seed: int, default: 0
        Random seed
    margin_frames: int, default: 0
        Margin in number of frames to avoid edge effects

    Returns
    -------
    chunk_list: np.array
        Array of concatenate chunks per segment
    """
    # TODO: if segment have differents length make another sampling that dependant on the length of the segment
    # Should be done by changing kwargs with total_num_chunks=XXX and total_duration=YYYY
    # And randomize the number of chunk per segment weighted by segment duration

    # check chunk size
    num_segments = recording.get_num_segments()
    for segment_index in range(num_segments):
        chunk_size_limit = recording.get_num_frames(segment_index) - 2 * margin_frames
        if chunk_size > chunk_size_limit:
            chunk_size = chunk_size_limit - 1
            warnings.warn(
                f"chunk_size is greater than the number "
                f"of samples for segment index {segment_index}. "
                f"Using {chunk_size}."
            )

    rng = np.random.default_rng(seed)
    chunk_list = []
    low = margin_frames
    size = num_chunks_per_segment
    for segment_index in range(num_segments):
        num_frames = recording.get_num_frames(segment_index)
        high = num_frames - chunk_size - margin_frames
        random_starts = rng.integers(low=low, high=high, size=size)
        segment_trace_chunk = [
            recording.get_traces(
                start_frame=start_frame,
                end_frame=(start_frame + chunk_size),
                segment_index=segment_index,
                return_scaled=return_scaled,
            )
            for start_frame in random_starts
        ]

        chunk_list.extend(segment_trace_chunk)

    if concatenated:
        return np.concatenate(chunk_list, axis=0)
    else:
        return chunk_list


def get_channel_distances(recording):
    """
    Distance between channel pairs
    """
    locations = recording.get_channel_locations()
    channel_distances = np.linalg.norm(locations[:, np.newaxis] - locations[np.newaxis, :], axis=2)

    return channel_distances


def get_closest_channels(recording, channel_ids=None, num_channels=None):
    """Get closest channels + distances

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to get closest channels
    channel_ids: list
        List of channels ids to compute there near neighborhood
    num_channels: int, default: None
        Maximum number of neighborhood channels to return

    Returns
    -------
    closest_channels_inds : array (2d)
        Closest channel indices in ascending order for each channel id given in input
    dists: array (2d)
        Distance in ascending order for each channel id given in input
    """
    if channel_ids is None:
        channel_ids = recording.get_channel_ids()
    if num_channels is None:
        num_channels = len(channel_ids) - 1

    locations = recording.get_channel_locations(channel_ids=channel_ids)

    closest_channels_inds = []
    dists = []
    for i in range(locations.shape[0]):
        distances = np.linalg.norm(locations[i, :] - locations, axis=1)
        order = np.argsort(distances)
        closest_channels_inds.append(order[1 : num_channels + 1])
        dists.append(distances[order][1 : num_channels + 1])

    return np.array(closest_channels_inds), np.array(dists)


def get_noise_levels(
    recording: "BaseRecording",
    return_scaled: bool = True,
    method: Literal["mad", "std"] = "mad",
    force_recompute: bool = False,
    **random_chunk_kwargs,
):
    """
    Estimate noise for each channel using MAD methods.
    You can use standard deviation with `method="std"`

    Internally it samples some chunk across segment.
    And then, it use MAD estimator (more robust than STD)

    Parameters
    ----------

    recording: BaseRecording
        The recording extractor to get noise levels
    return_scaled: bool
        If True, returned noise levels are scaled to uV
    method: "mad" | "std", default: "mad"
        The method to use to estimate noise levels
    force_recompute: bool
        If True, noise levels are recomputed even if they are already stored in the recording extractor
    random_chunk_kwargs: dict
        Kwargs for get_random_data_chunks

    Returns
    -------
    noise_levels: array
        Noise levels for each channel
    """

    if return_scaled:
        key = f"noise_level_{method}_scaled"
    else:
        key = f"noise_level_{method}_raw"

    if key in recording.get_property_keys() and not force_recompute:
        noise_levels = recording.get_property(key=key)
    else:
        random_chunks = get_random_data_chunks(recording, return_scaled=return_scaled, **random_chunk_kwargs)

        if method == "mad":
            med = np.median(random_chunks, axis=0, keepdims=True)
            # hard-coded so that core doesn't depend on scipy
            noise_levels = np.median(np.abs(random_chunks - med), axis=0) / 0.6744897501960817
        elif method == "std":
            noise_levels = np.std(random_chunks, axis=0)
        recording.set_property(key, noise_levels)

    return noise_levels


def get_chunk_with_margin(
    rec_segment,
    start_frame,
    end_frame,
    channel_indices,
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
    length = rec_segment.get_num_samples()

    if channel_indices is None:
        channel_indices = slice(None)

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

        traces_chunk = rec_segment.get_traces(
            start_frame - left_margin,
            end_frame + right_margin,
            channel_indices,
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

        traces_chunk = rec_segment.get_traces(start_frame2, end_frame2, channel_indices)

        if dtype is not None or window_on_margin or left_pad > 0 or right_pad > 0:
            need_copy = True
        else:
            need_copy = False

        left_margin = margin
        right_margin = margin

        if need_copy:
            if dtype is None:
                dtype = traces_chunk.dtype

            left_margin = margin
            if end_frame < (length + margin):
                right_margin = margin
            else:
                right_margin = end_frame + margin - length

            if add_zeros:
                traces_chunk2 = np.zeros((full_size, traces_chunk.shape[1]), dtype=dtype)
                i0 = left_pad
                i1 = left_pad + traces_chunk.shape[0]
                traces_chunk2[i0:i1, :] = traces_chunk
                if window_on_margin:
                    # apply inplace taper on border
                    taper = (1 - np.cos(np.arange(margin) / margin * np.pi)) / 2
                    taper = taper[:, np.newaxis]
                    traces_chunk2[:margin] *= taper
                    traces_chunk2[-margin:] *= taper[::-1]
                traces_chunk = traces_chunk2
            elif add_reflect_padding:
                # in this case, we don't want to taper
                traces_chunk = np.pad(
                    traces_chunk.astype(dtype, copy=False),
                    [(left_pad, right_pad), (0, 0)],
                    mode="reflect",
                )
            else:
                # we need a copy to change the dtype
                traces_chunk = np.asarray(traces_chunk, dtype=dtype)

    return traces_chunk, left_margin, right_margin


def order_channels_by_depth(recording, channel_ids=None, dimensions=("x", "y"), flip=False):
    """
    Order channels by depth, by first ordering the x-axis, and then the y-axis.

    Parameters
    ----------
    recording : BaseRecording
        The input recording
    channel_ids : list/array or None
        If given, a subset of channels to order locations for
    dimensions : str, tuple, or list, default: ('x', 'y')
        If str, it needs to be 'x', 'y', 'z'.
        If tuple or list, it sorts the locations in two dimensions using lexsort.
        This approach is recommended since there is less ambiguity
    flip: bool, default: False
        If flip is False then the order is bottom first (starting from tip of the probe).
        If flip is True then the order is upper first.

    Returns
    -------
    order_f : np.array
        Array with sorted indices
    order_r : np.array
        Array with indices to revert sorting
    """
    locations = recording.get_channel_locations()
    ndim = locations.shape[1]
    channel_inds = recording.ids_to_indices(ids=channel_ids, prefer_slice=True)
    locations = locations[channel_inds, :]

    if isinstance(dimensions, str):
        dim = ["x", "y", "z"].index(dimensions)
        assert dim < ndim, "Invalid dimensions!"
        order_f = np.argsort(locations[:, dim], kind="stable")
    else:
        assert isinstance(dimensions, (tuple, list)), "dimensions can be str, tuple, or list"
        locations_to_sort = ()
        for dim in dimensions:
            dim = ["x", "y", "z"].index(dim)
            assert dim < ndim, "Invalid dimensions!"
            locations_to_sort += (locations[:, dim],)
        order_f = np.lexsort(locations_to_sort)
    if flip:
        order_f = order_f[::-1]
    order_r = np.argsort(order_f, kind="stable")

    return order_f, order_r


def check_probe_do_not_overlap(probes):
    """
    When several probes this check that that they do not overlap in space
    and so channel positions can be safly concatenated.
    """
    for i in range(len(probes)):
        probe_i = probes[i]
        # check that all positions in probe_j are outside probe_i boundaries
        x_bounds_i = [
            np.min(probe_i.contact_positions[:, 0]),
            np.max(probe_i.contact_positions[:, 0]),
        ]
        y_bounds_i = [
            np.min(probe_i.contact_positions[:, 1]),
            np.max(probe_i.contact_positions[:, 1]),
        ]

        for j in range(i + 1, len(probes)):
            probe_j = probes[j]

            if np.any(
                np.array(
                    [
                        x_bounds_i[0] < cp[0] < x_bounds_i[1] and y_bounds_i[0] < cp[1] < y_bounds_i[1]
                        for cp in probe_j.contact_positions
                    ]
                )
            ):
                raise Exception("Probes are overlapping! Retrieve locations of single probes separately")


def get_rec_attributes(recording):
    """
    Construct rec_attributes from recording object

    Parameters
    ----------
    recording : BaseRecording
        The recording object

    Returns
    -------
    dict
        The rec_attributes dictionary
    """
    properties_to_attrs = deepcopy(recording._properties)
    if "contact_vector" in properties_to_attrs:
        del properties_to_attrs["contact_vector"]
    rec_attributes = dict(
        channel_ids=recording.channel_ids,
        sampling_frequency=recording.get_sampling_frequency(),
        num_channels=recording.get_num_channels(),
        num_samples=[recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())],
        is_filtered=recording.is_filtered(),
        properties=properties_to_attrs,
        dtype=recording.get_dtype(),
    )
    return rec_attributes
