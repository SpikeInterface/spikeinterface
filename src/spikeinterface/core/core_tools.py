from pathlib import Path
from typing import Union
import os
import sys
import datetime
import json
from copy import deepcopy
import gc
import mmap
import inspect

import numpy as np
from tqdm import tqdm

from .job_tools import (
    ensure_chunk_size,
    ensure_n_jobs,
    divide_segment_into_chunks,
    fix_job_kwargs,
    ChunkRecordingExecutor,
    _shared_job_kwargs_doc,
)


def define_function_from_class(source_class, name):
    "Wrapper to change the name of a class"

    return source_class


def read_python(path):
    """Parses python scripts in a dictionary

    Parameters
    ----------
    path: str or Path
        Path to file to parse

    Returns
    -------
    metadata:
        dictionary containing parsed file

    """
    from six import exec_
    import re

    path = Path(path).absolute()
    assert path.is_file()
    with path.open("r") as f:
        contents = f.read()
    contents = re.sub(r"range\(([\d,]*)\)", r"list(range(\1))", contents)
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def write_python(path, dict):
    """Saves python dictionary to file

    Parameters
    ----------
    path: str or Path
        Path to save file
    dict: dict
        dictionary to save
    """
    with Path(path).open("w") as f:
        for k, v in dict.items():
            if isinstance(v, str) and not v.startswith("'"):
                if "path" in k and "win" in sys.platform:
                    f.write(str(k) + " = r'" + str(v) + "'\n")
                else:
                    f.write(str(k) + " = '" + str(v) + "'\n")
            else:
                f.write(str(k) + " = " + str(v) + "\n")


class SIJsonEncoder(json.JSONEncoder):
    """
    An encoder used to encode Spike interface objects to json
    """

    def default(self, obj):
        from spikeinterface.core.base import BaseExtractor

        # Over-write behaviors for datetime object
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()

        # This should transforms integer, floats and bool to their python counterparts
        if isinstance(obj, np.generic):
            return obj.item()

        if np.issctype(obj):  # Cast numpy datatypes to their names
            return np.dtype(obj).name

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, BaseExtractor):
            return obj.to_dict()

        # The base-class handles the assertion
        return super().default(obj)

    # This machinery is necessary for overriding the default behavior of the json encoder with keys
    # This is a deep issue that goes deep down to cpython: https://github.com/python/cpython/issues/63020
    # This object is called before encoding (so it pre-processes the object to not have numpy scalars)
    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self.remove_numpy_scalars(obj), _one_shot=_one_shot)

    def remove_numpy_scalars(self, object):
        from spikeinterface.core.base import BaseExtractor

        if isinstance(object, dict):
            return self.remove_numpy_scalars_in_dict(object)
        elif isinstance(object, (list, tuple, set)):
            return self.remove_numpy_scalars_in_list(object)
        elif isinstance(object, BaseExtractor):
            return self.remove_numpy_scalars_in_dict(object.to_dict())
        else:
            return object.item() if isinstance(object, np.generic) else object

    def remove_numpy_scalars_in_list(self, list_: Union[list, tuple, set]) -> list:
        return [self.remove_numpy_scalars(obj) for obj in list_]

    def remove_numpy_scalars_in_dict(self, dictionary: dict) -> dict:
        dict_copy = dict()
        for key, value in dictionary.items():
            key = self.remove_numpy_scalars(key)
            value = self.remove_numpy_scalars(value)
            dict_copy[key] = value

        return dict_copy


def check_json(dictionary: dict) -> dict:
    """
    Function that transforms a dictionary with spikeinterface objects into a json writable dictionary

    Parameters
    ----------
    dictionary : A dictionary

    """

    json_string = json.dumps(dictionary, indent=4, cls=SIJsonEncoder)
    return json.loads(json_string)


def add_suffix(file_path, possible_suffix):
    file_path = Path(file_path)
    if isinstance(possible_suffix, str):
        possible_suffix = [possible_suffix]
    possible_suffix = [s if s.startswith(".") else "." + s for s in possible_suffix]
    if file_path.suffix not in possible_suffix:
        file_path = file_path.parent / (file_path.name + "." + possible_suffix[0])
    return file_path


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
    time_axis: 0 (default) or 1
        If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
        If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
    offset: int
        number of offset bytes

    """
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
        traces = traces.astype(dtype)
    array[start_frame:end_frame, :] = traces

    # Close the memmap
    memmap_obj.flush()


def write_binary_recording(
    recording,
    file_paths=None,
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
    file_path: str
        The path to the file.
    dtype: dtype
        Type of the saved data. Default float32.
    add_file_extension: bool
        If True (default), file the '.raw' file extension is added if the file name is not a 'raw', 'bin', or 'dat'
    byte_offset: int
        Offset in bytes (default 0) to for the binary file (e.g. to write a header)
    auto_cast_uint: bool
        If True (default), unsigned integers are automatically cast to int if the specified dtype is signed
    {}
    """
    assert file_paths is not None, "Provide 'file_path'"
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
            traces = traces.astype(dtype)
        traces.tofile(file_handle)
    else:
        num_frames = recording.get_num_samples(segment_index=0)
        chunks = divide_segment_into_chunks(num_frames, chunk_size)

        for start_frame, end_frame in chunks:
            traces = recording.get_traces(segment_index=0, start_frame=start_frame, end_frame=end_frame)
            if time_axis == 1:
                traces = traces.T
            if dtype is not None:
                traces = traces.astype(dtype)
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
    traces = traces.astype(dtype)
    arr[start_frame:end_frame, :] = traces


def make_shared_array(shape, dtype):
    from multiprocessing.shared_memory import SharedMemory

    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape) * dtype.itemsize)
    shm = SharedMemory(name=None, create=True, size=nbytes)
    arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    arr[:] = 0

    return arr, shm


def write_memory_recording(recording, dtype=None, verbose=False, auto_cast_uint=True, **job_kwargs):
    """
    Save the traces into numpy arrays (memory).
    try to use the SharedMemory introduce in py3.8 if n_jobs > 1

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    dtype: dtype
        Type of the saved data. Default float32.
    verbose: bool
        If True, output is verbose (when chunks are used)
    auto_cast_uint: bool
        If True (default), unsigned integers are automatically cast to int if the specified dtype is signed
    {}

    Returns
    ---------
    arrays: one arrays per segment
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
    shapes = []

    n_jobs = ensure_n_jobs(recording, n_jobs=job_kwargs.get("n_jobs", 1))
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        num_channels = recording.get_num_channels()
        shape = (num_frames, num_channels)
        shapes.append(shape)
        if n_jobs > 1:
            arr, shm = make_shared_array(shape, dtype)
            shm_names.append(shm.name)
        else:
            arr = np.zeros(shape, dtype=dtype)
        arrays.append(arr)

    # use executor (loop or workers)
    func = _write_memory_chunk
    init_func = _init_memory_worker
    if n_jobs > 1:
        init_args = (recording, None, shm_names, shapes, dtype, cast_unsigned)
    else:
        init_args = (recording, arrays, None, None, dtype, cast_unsigned)

    executor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, verbose=verbose, job_name="write_memory_recording", **job_kwargs
    )
    executor.run()

    return arrays


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
        Path to dataset in h5 file (e.g. '/dataset')
    segment_index: int
        index of segment
    save_path: str
        The path to the file.
    file_handle: file handle
        The file handle to dump data. This can be used to append data to an header. In case file_handle is given,
        the file is NOT closed after writing the binary data.
    time_axis: 0 (default) or 1
        If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
        If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
    single_axis: bool, default False
        If True, a single-channel recording is saved as a one dimensional array.
    dtype: dtype
        Type of the saved data. Default float32.
    chunk_size: None or int
        Number of chunks to save the file in. This avoid to much memory consumption for big files.
        If None and 'chunk_memory' is given, the file is saved in chunks of 'chunk_memory' MB (default 500MB)
    chunk_memory: None or str
        Chunk size in bytes must endswith 'k', 'M' or 'G' (default '500M')
    verbose: bool
        If True, output is verbose (when chunks are used)
    auto_cast_uint: bool
        If True (default), unsigned integers are automatically cast to int if the specified dtype is signed
    return_scaled : bool, optional
        If True and the recording has scaling (gain_to_uV and offset_to_uV properties),
        traces are dumped to uV, by default False
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
            traces = traces.astype(dtype_file)
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
                traces = traces.astype(dtype_file)
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


def write_traces_to_zarr(
    recording,
    zarr_root,
    zarr_path,
    storage_options,
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
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    zarr_root: zarr.Group
        The zarr root
    zarr_path: str or Path
        The path to the zarr file
    storage_options: dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.
    dataset_paths: list
        List of paths to traces datasets in the zarr group
    channel_chunk_size: int or None
        Channels per chunk. Default None (chunking in time only)
    dtype: dtype
        Type of the saved data. Default float32.
    compressor: zarr compressor or None
        Zarr compressor
    filters: list
        List of zarr filters
    verbose: bool
        If True, output is verbose (when chunks are used)
    auto_cast_uint: bool
        If True (default), unsigned integers are automatically cast to int if the specified dtype is signed
    {}
    """
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
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        num_channels = recording.get_num_channels()
        dset_name = dataset_paths[segment_index]
        shape = (num_frames, num_channels)
        _ = zarr_root.create_dataset(
            name=dset_name,
            shape=shape,
            chunks=(chunk_size, channel_chunk_size),
            dtype=dtype,
            filters=filters,
            compressor=compressor,
        )
        # synchronizer=zarr.ThreadSynchronizer())

    # use executor (loop or workers)
    func = _write_zarr_chunk
    init_func = _init_zarr_worker
    init_args = (recording, zarr_path, storage_options, dataset_paths, dtype, cast_unsigned)
    executor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, verbose=verbose, job_name="write_zarr_recording", **job_kwargs
    )
    executor.run()


# used by write_zarr_recording + ChunkRecordingExecutor
def _init_zarr_worker(recording, zarr_path, storage_options, dataset_paths, dtype, cast_unsigned):
    import zarr

    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor

        worker_ctx["recording"] = load_extractor(recording)
    else:
        worker_ctx["recording"] = recording

    # reload root and datasets
    if storage_options is None:
        if isinstance(zarr_path, str):
            zarr_path_init = zarr_path
            zarr_path = Path(zarr_path)
        else:
            zarr_path_init = str(zarr_path)
    else:
        zarr_path_init = zarr_path

    root = zarr.open(zarr_path_init, mode="r+", storage_options=storage_options)
    zarr_datasets = []
    for dset_name in dataset_paths:
        z = root[dset_name]
        zarr_datasets.append(z)
    worker_ctx["zarr_datasets"] = zarr_datasets
    worker_ctx["dtype"] = np.dtype(dtype)
    worker_ctx["cast_unsigned"] = cast_unsigned

    return worker_ctx


# used by write_zarr_recording + ChunkRecordingExecutor
def _write_zarr_chunk(segment_index, start_frame, end_frame, worker_ctx):
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


def determine_cast_unsigned(recording, dtype):
    recording_dtype = np.dtype(recording.get_dtype())

    if np.dtype(dtype) != recording_dtype and recording_dtype.kind == "u" and np.dtype(dtype).kind == "i":
        cast_unsigned = True
    else:
        cast_unsigned = False
    return cast_unsigned


def is_dict_extractor(d):
    """
    Check if a dict describe an extractor.
    """
    if not isinstance(d, dict):
        return False
    is_extractor = ("module" in d) and ("class" in d) and ("version" in d) and ("annotations" in d)
    return is_extractor


def recursive_path_modifier(d, func, target="path", copy=True) -> dict:
    """
    Generic function for recursive modification of paths in an extractor dict.
    A recording can be nested and this function explores the dictionary recursively
    to find the parent file or folder paths.

    Useful for :
      * relative/absolute path change
      * docker rebase path change

    Modification is inplace with an optional copy.

    Parameters
    ----------
    d : dict
        Extractor dictionary
    func : function
        Function to apply to the path. It must take a path as input and return a path
    target : str, optional
        String to match to dictionary key, by default 'path'
    copy : bool, optional
        If True the original dictionary is deep copied, by default True (at first call)

    Returns
    -------
    dict
        Modified dictionary
    """
    if copy:
        dc = deepcopy(d)
    else:
        dc = d

    if "kwargs" in dc.keys():
        kwargs = dc["kwargs"]

        # change in place (copy=False)
        recursive_path_modifier(kwargs, func, copy=False)

        # find nested and also change inplace (copy=False)
        nested_extractor_dict = None
        for k, v in kwargs.items():
            if isinstance(v, dict) and is_dict_extractor(v):
                nested_extractor_dict = v
                recursive_path_modifier(nested_extractor_dict, func, copy=False)
            # deal with list of extractor objects (e.g. concatenate_recordings)
            elif isinstance(v, list):
                for vl in v:
                    if isinstance(vl, dict) and is_dict_extractor(vl):
                        nested_extractor_dict = vl
                        recursive_path_modifier(nested_extractor_dict, func, copy=False)

        return dc
    else:
        for k, v in d.items():
            if target in k:
                # paths can be str or list of str or None
                if v is None:
                    continue
                if isinstance(v, (str, Path)):
                    dc[k] = func(v)
                elif isinstance(v, list):
                    dc[k] = [func(e) for e in v]
                else:
                    raise ValueError(f"{k} key for path  must be str or list[str]")


def recursive_key_finder(d, key):
    # Find all values for a key on a dictionary, even if nested
    for k, v in d.items():
        if isinstance(v, dict):
            yield from recursive_key_finder(v, key)
        else:
            if k == key:
                yield v


def convert_seconds_to_str(seconds: float, long_notation: bool = True) -> str:
    """
    Convert seconds to a human-readable string representation.
    Parameters
    ----------
    seconds : float
        The duration in seconds.
    long_notation : bool, optional, default: True
        Whether to display the time with additional units (such as milliseconds, minutes,
        hours, or days). If set to True, the function will display a more detailed
        representation of the duration, including other units alongside the primary
        seconds representation.
    Returns
    -------
    str
        A string representing the duration, with additional units included if
        requested by the `long_notation` parameter.
    """
    base_str = f"{seconds:,.2f}s"

    if long_notation:
        if seconds < 1.0:
            base_str += f" ({seconds * 1000:.2f} ms)"
        elif seconds < 60:
            pass  # seconds is already the primary representation
        elif seconds < 3600:
            minutes = seconds / 60
            base_str += f" ({minutes:.2f} minutes)"
        elif seconds < 86400 * 2:  # 2 days
            hours = seconds / 3600
            base_str += f" ({hours:.2f} hours)"
        else:
            days = seconds / 86400
            base_str += f" ({days:.2f} days)"

    return base_str


def convert_bytes_to_str(byte_value: int) -> str:
    """
    Convert a number of bytes to a human-readable string with an appropriate unit.

    This function converts a given number of bytes into a human-readable string
    representing the value in either bytes (B), kibibytes (KiB), mebibytes (MiB),
    gibibytes (GiB), or tebibytes (TiB). The function uses the IEC binary prefixes
    (1 KiB = 1024 B, 1 MiB = 1024 KiB, etc.) to determine the appropriate unit.

    Parameters
    ----------
    byte_value : int
        The number of bytes to convert.

    Returns
    -------
    str
        The converted value as a formatted string with two decimal places,
        followed by a space and the appropriate unit (B, KiB, MiB, GiB, or TiB).

    Examples
    --------
    >>> convert_bytes_to_str(1024)
    '1.00 KiB'
    >>> convert_bytes_to_str(1048576)
    '1.00 MiB'
    >>> convert_bytes_to_str(45056)
    '43.99 KiB'
    """
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while byte_value >= 1024 and i < len(suffixes) - 1:
        byte_value /= 1024
        i += 1
    return f"{byte_value:.2f} {suffixes[i]}"
