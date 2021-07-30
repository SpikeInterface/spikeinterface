from pathlib import Path
import os
import datetime

import numpy as np
from tqdm import tqdm

from .job_tools import ensure_chunk_size, ensure_n_jobs, divide_segment_into_chunks, ChunkRecordingExecutor, \
    _shared_job_kwargs_doc


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
    with path.open('r') as f:
        contents = f.read()
    contents = re.sub(r'range\(([\d,]*)\)', r'list(range(\1))', contents)
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
    with Path(path).open('w') as f:
        for k, v in dict.items():
            if isinstance(v, str) and not v.startswith("'"):
                if 'path' in k and 'win' in sys.platform:
                    f.write(str(k) + " = r'" + str(v) + "'\n")
                else:
                    f.write(str(k) + " = '" + str(v) + "'\n")
            else:
                f.write(str(k) + " = " + str(v) + "\n")


def check_json(d):
    # quick hack to ensure json writable
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = check_json(v)
        elif isinstance(v, Path):
            d[k] = str(v.absolute())
        elif isinstance(v, bool):
            d[k] = bool(v)
        elif isinstance(v, (np.int32, np.int64)):
            d[k] = int(v)
        elif isinstance(v, (np.float32, np.float64)):
            d[k] = float(v)
        elif isinstance(v, datetime.datetime):
            d[k] = v.isoformat()
        elif isinstance(v, (np.ndarray, list)):
            if len(v) > 0:
                if isinstance(v[0], dict):
                    # these must be extractors for multi extractors
                    d[k] = [check_json(v_el) for v_el in v]
                else:
                    v_arr = np.array(v)
                    if len(v_arr.shape) == 1:
                        if 'int' in str(v_arr.dtype):
                            v_arr = [int(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        elif 'float' in str(v_arr.dtype):
                            v_arr = [float(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        elif isinstance(v_arr[0], str):
                            v_arr = [str(v_el) for v_el in v_arr]
                            d[k] = v_arr
                        else:
                            print(f'Skipping field {k}: only 1D arrays of int, float, or str types can be serialized')
                    elif len(v_arr.shape) == 2:
                        if 'int' in str(v_arr.dtype):
                            v_arr = [[int(v_el) for v_el in v_row] for v_row in v_arr]
                            d[k] = v_arr
                        elif 'float' in str(v_arr.dtype):
                            v_arr = [[float(v_el) for v_el in v_row] for v_row in v_arr]
                            d[k] = v_arr
                        else:
                            print(f'Skipping field {k}: only 2D arrays of int or float type can be serialized')
                    else:
                        print(f"Skipping field {k}: only 1D and 2D arrays can be serialized")
            else:
                d[k] = list(v)
    return d


def add_suffix(file_path, possible_suffix):
    file_path = Path(file_path)
    if isinstance(possible_suffix, str):
        possible_suffix = [possible_suffix]
    possible_suffix = [s if s.startswith('.') else '.' + s for s in possible_suffix]
    if file_path.suffix not in possible_suffix:
        file_path = file_path.parent / (file_path.name + '.' + possible_suffix[0])
    return file_path


def read_binary_recording(file, num_chan, dtype, time_axis=0, offset=0):
    '''
    Read binary .bin or .dat file.

    Parameters
    ----------
    file: str
        File name
    num_chan: int
        Number of channels
    dtype: dtype
        dtype of the file
    time_axis: 0 (default) or 1
        If 0 then traces are transposed to ensure (nb_sample, nb_channel) in the file.
        If 1, the traces shape (nb_channel, nb_sample) is kept in the file.
    offset: int
        number of offset bytes

    '''
    num_chan = int(num_chan)
    with Path(file).open() as f:
        nsamples = (os.fstat(f.fileno()).st_size - offset) // (num_chan * np.dtype(dtype).itemsize)
    if time_axis == 0:
        samples = np.memmap(file, np.dtype(dtype), mode='r', offset=offset, shape=(nsamples, num_chan))
    else:
        samples = np.memmap(file, np.dtype(dtype), mode='r', offset=offset, shape=(num_chan, nsamples)).T
    return samples


# used by write_binary_recording + ChunkRecordingExecutor
def _init_binary_worker(recording, rec_memmaps_dict, dtype):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        worker_ctx['recording'] = load_extractor(recording)
    else:
        worker_ctx['recording'] = recording

    rec_memmaps = []
    for d in rec_memmaps_dict:
        rec_memmaps.append(np.memmap(**d))

    worker_ctx['rec_memmaps'] = rec_memmaps
    worker_ctx['dtype'] = np.dtype(dtype)

    return worker_ctx


# used by write_binary_recording + ChunkRecordingExecutor
def _write_binary_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    dtype = worker_ctx['dtype']
    rec_memmap = worker_ctx['rec_memmaps'][segment_index]

    # apply function
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    traces = traces.astype(dtype)
    rec_memmap[start_frame:end_frame, :] = traces


def write_binary_recording(recording, file_paths=None, dtype=None, add_file_extension=True,
                           verbose=False, byte_offset=0, **job_kwargs):
    '''
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
    verbose: bool
        If True, output is verbose (when chunks are used)
    byte_offset: int
        Offset in bytes (default 0) to for the binary file (e.g. to write a header)
    {}
    '''
    assert file_paths is not None, "Provide 'file_path'"

    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    file_paths = [Path(e) for e in file_paths]
    if add_file_extension:
        file_paths = [add_suffix(file_path, ['raw', 'bin', 'dat']) for file_path in file_paths]

    if dtype is None:
        dtype = recording.get_dtype()

    # create memmap files
    rec_memmaps = []
    rec_memmaps_dict = []
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        num_channels = recording.get_num_channels()
        file_path = file_paths[segment_index]
        shape = (num_frames, num_channels)
        rec_memmap = np.memmap(str(file_path), dtype=dtype, mode='w+', offset=byte_offset, shape=shape)
        rec_memmaps.append(rec_memmap)
        rec_memmaps_dict.append(dict(filename=str(file_path), dtype=dtype, mode='r+', offset=byte_offset, shape=shape))

    # use executor (loop or workers)
    func = _write_binary_chunk
    init_func = _init_binary_worker
    init_args = (recording.to_dict(), rec_memmaps_dict, dtype)
    executor = ChunkRecordingExecutor(recording, func, init_func, init_args, verbose=verbose,
                                      job_name='write_binary_recording', **job_kwargs)
    executor.run()


write_binary_recording.__doc__ = write_binary_recording.__doc__.format(_shared_job_kwargs_doc)


def write_binary_recording_file_handle(recording, file_handle=None,
                                       time_axis=0, dtype=None, byte_offset=0, verbose=False, **job_kwargs):
    """
    Old variant version of write_binary_recording with one file handle.
    Can be usefull in some case ???
    Not used naymore at the moment.

    @ SAM useful for writing with time_axis=1!
    """
    assert file_handle is not None
    assert recording.get_num_segments() == 1, 'If file_handle is given then only deals with one segment'

    if dtype is None:
        dtype = recording.get_dtype()

    chunk_size = ensure_chunk_size(recording, **job_kwargs)

    if chunk_size is not None and time_axis == 1:
        print("Chunking disabled due to 'time_axis' == 1")
        chunk_size = None

    if chunk_size is None:
        # no chunkking
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
            traces = recording.get_traces(segment_index=0,
                                          start_frame=start_frame, end_frame=end_frame)
            if time_axis == 1:
                traces = traces.T
            if dtype is not None:
                traces = traces.astype(dtype)
            file_handle.write(traces.tobytes())


# used by write_memory_recording
def _init_memory_worker(recording, arrays, shm_names, shapes, dtype):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        worker_ctx['recording'] = load_extractor(recording)
    else:
        worker_ctx['recording'] = recording

    worker_ctx['dtype'] = np.dtype(dtype)

    if arrays is None:
        # create it from share memory name
        from multiprocessing.shared_memory import SharedMemory
        arrays = []
        # keep shm alive
        worker_ctx['shms'] = []
        for i in range(len(shm_names)):
            shm = SharedMemory(shm_names[i])
            worker_ctx['shms'].append(shm)
            arr = np.ndarray(shape=shapes[i], dtype=dtype, buffer=shm.buf)
            arrays.append(arr)

    worker_ctx['arrays'] = arrays

    return worker_ctx


# used by write_memory_recording
def _write_memory_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    dtype = worker_ctx['dtype']
    arr = worker_ctx['arrays'][segment_index]

    # apply function
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    traces = traces.astype(dtype)
    arr[start_frame:end_frame, :] = traces


def make_shared_array(shape, dtype):
    # https://docs.python.org/3/library/multiprocessing.shared_memory.html
    try:
        from multiprocessing.shared_memory import SharedMemory
    except Exception as e:
        raise Exception('SharedMemory is available only for python>=3.8')

    dtype = np.dtype(dtype)
    nbytes = shape[0] * shape[1] * dtype.itemsize
    shm = SharedMemory(name=None, create=True, size=nbytes)
    arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    arr[:] = 0

    return arr, shm


def write_memory_recording(recording, dtype=None, verbose=False, **job_kwargs):
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
    {}

    Returns
    ---------
    arrays: one arrays per segment
    """

    chunk_size = ensure_chunk_size(recording, **job_kwargs)
    n_jobs = ensure_n_jobs(recording, n_jobs=job_kwargs.get('n_jobs', 1))

    if dtype is None:
        dtype = recording.get_dtype()

    # create sharedmmep
    arrays = []
    shm_names = []
    shapes = []
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
        init_args = (recording.to_dict(), None, shm_names, shapes, dtype)
    else:
        init_args = (recording.to_dict(), arrays, None, None, dtype)

    executor = ChunkRecordingExecutor(recording, func, init_func, init_args, verbose=verbose,
                                      job_name='write_memory_recording', **job_kwargs)
    executor.run()

    return arrays


write_memory_recording.__doc__ = write_memory_recording.__doc__.format(_shared_job_kwargs_doc)


def write_to_h5_dataset_format(recording, dataset_path, segment_index, save_path=None, file_handle=None,
                               time_axis=0, single_axis=False, dtype=None, chunk_size=None, chunk_memory='500M',
                               verbose=False):
    """
    Save the traces of a recording extractor in an h5 dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    dataset_path: str
        Path to dataset in h5 filee (e.g. '/dataset')
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
    """
    import h5py
    # ~ assert HAVE_H5, "To write to h5 you need to install h5py: pip install h5py"
    assert save_path is not None or file_handle is not None, "Provide 'save_path' or 'file handle'"

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == '':
            # when suffix is already raw/bin/dat do not change it.
            save_path = save_path.parent / (save_path.name + '.h5')

    num_channels = recording.get_num_channels()
    num_frames = recording.get_num_frames(segment_index=0)

    if file_handle is not None:
        assert isinstance(file_handle, h5py.File)
    else:
        file_handle = h5py.File(save_path, 'w')

    if dtype is None:
        dtype_file = recording.get_dtype()
    else:
        dtype_file = dtype

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
        traces = recording.get_traces()
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
            traces = recording.get_traces(segment_index=segment_index,
                                          start_frame=i * chunk_size,
                                          end_frame=min((i + 1) * chunk_size, num_frames))
            chunk_frames = traces.shape[0]
            if dtype is not None:
                traces = traces.astype(dtype_file)
            if single_axis:
                dset[chunk_start:chunk_start + chunk_frames] = traces[:, 0]
            else:
                if time_axis == 0:
                    dset[chunk_start:chunk_start + chunk_frames, :] = traces
                else:
                    dset[:, chunk_start:chunk_start + chunk_frames] = traces.T

            chunk_start += chunk_frames

    if save_path is not None:
        file_handle.close()
    return save_path
