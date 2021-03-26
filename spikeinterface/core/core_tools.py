from pathlib import Path
import os
import datetime

import numpy as np

from joblib import Parallel, delayed

from . job_tools import ensure_chunk_size, ensure_n_jobs, ChunkRecordingExecutor


def check_json(d):
    # quick hack to ensure json writable
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = check_json(v)
        elif isinstance(v, Path):
            d[k] = str(v.absolute())
        elif isinstance(v, bool):
            d[k] = bool(v)
        elif isinstance(v, (np.int, np.int32, np.int64)):
            d[k] = int(v)
        elif isinstance(v, (np.float, np.float32, np.float64)):
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
    possible_suffix = [s if s.startswith('.') else '.' + s for s in possible_suffix ]
    if file_path.suffix not in possible_suffix:
        file_path = file_path.parent / (file_path.name + '.' + possible_suffix[0])
    return file_path


def read_binary_recording(file, num_chan, dtype, time_axis=0, offset=0):
    '''
    Reads binary .bin or .dat file.

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



# used by write_binary_recording
def _init_binary_worker(recording, rec_memmaps, dtype):
    # create a local dict per worker
    local_dict = {}
    from spikeinterface.core import load_extractor
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        local_dict['recording'] = load_extractor(recording)
    else:
        local_dict['recording'] = recording
    
    local_dict['rec_memmaps'] = rec_memmaps
    local_dict['dtype'] = np.dtype(dtype)
    
    return local_dict


# used by write_binary_recording
def _write_binary_chunk(segment_index, start_frame, end_frame, local_dict):
    # recover variables of the worker
    recording = local_dict['recording']
    dtype = local_dict['dtype']
    rec_memmap = local_dict['rec_memmaps'][segment_index]
    
    # apply function
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    traces = traces.astype(dtype)
    rec_memmap[start_frame:end_frame, :] = traces
    



def write_binary_recording(recording, files_path=None, dtype=None, 
                                                    verbose=False, **job_kwargs):
    '''Saves the traces of a recording extractor in several binary .dat format.

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
    verbose: bool
        If True, output is verbose (when chunks are used)

    **job_kwargs: 
        Use by job_tools modules to set:
            * chunk_size or chunk_memory, or total_memory
            * n_jobs
            * progress_bar 
    '''
    assert files_path is not None , "Provide 'file_path'"
    
    if not isinstance(files_path, list):
        files_path = [files_path]
    files_path = [Path(e) for e in files_path]
    files_path = [add_suffix(file_path, ['raw', 'bin', 'dat']) for file_path in files_path]
    
    if dtype is None:
        dtype = recording.get_dtype()

    # create memmap files
    rec_memmaps = []
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        num_channels = recording.get_num_channels()
        file_path = files_path[segment_index]
        shape = (num_frames, num_channels)
        rec_memmap = np.memmap(str(file_path), dtype=dtype, mode='w+', shape=shape)
        rec_memmaps.append(rec_memmap)
    
    # use executor (loop or workers)
    func = _write_binary_chunk
    init_func = _init_binary_worker
    init_args = (recording.to_dict(), rec_memmaps, dtype)
    executor = ChunkRecordingExecutor(recording, func, init_func, init_args, verbose=verbose,
                    job_name='write_binary_recording', **job_kwargs)
    executor.run()


def write_binary_recording_file_handle(recording, file_handle=None,
                               time_axis=0, dtype=None, verbose=False, **job_kwargs):
    """
    Old variant version of write_binary_recording with one file handle.
    Can be usefull in some case ???
    Not used naymore at the moment.
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
        chunks = divide_into_chunks(num_frames, chunk_size)
        
        for start_frame, end_frame in chunks:
            traces = recording.get_traces(segment_index=0, 
                        start_frame=start_frame, end_frame=end_frame)
            if time_axis == 1:
                traces = traces.T
            if dtype is not None:
                traces = traces.astype(dtype)
            file_handle.write(traces.tobytes())






def write_to_h5_dataset_format(recording, dataset_path, segment_index, save_path=None, file_handle=None,
                               time_axis=0, dtype=None, chunk_size=None, chunk_mb=500, verbose=False):
    '''Saves the traces of a recording extractor in an h5 dataset.

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
    dtype: dtype
        Type of the saved data. Default float32.
    chunk_size: None or int
        Number of chunks to save the file in. This avoid to much memory consumption for big files.
        If None and 'chunk_mb' is given, the file is saved in chunks of 'chunk_mb' Mb (default 500Mb)
    chunk_mb: None or int
        Chunk size in Mb (default 500Mb)
    verbose: bool
        If True, output is verbose (when chunks are used)
    '''
    import h5py
    #~ assert HAVE_H5, "To write to h5 you need to install h5py: pip install h5py"
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

    if time_axis == 0:
        shape= (num_frames, num_channels)
    else:
        shape= (num_channels, num_frames)
    dset = file_handle.create_dataset(dataset_path, shape=shape, dtype=dtype_file)
    
    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (num_channels * n_bytes)

    if chunk_size is None:
        traces = recording.get_traces()
        if dtype is not None:
            traces = traces.astype(dtype_file)
        if time_axis == 1:
            traces = traces.T
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
            traces = recording.get_traces(start_frame=i * chunk_size,
                                          end_frame=min((i + 1) * chunk_size, num_frames))
            chunk_frames = traces.shape[0]
            if dtype is not None:
                traces = traces.astype(dtype_file)
            if time_axis == 0:
                dset[chunk_start:chunk_start + chunk_frames, :] = traces
            else:
                dset[:, chunk_start:chunk_start + chunk_frames] = traces.T
            chunk_start += chunk_frames

    if save_path is not None:
        file_handle.close()
    return save_path


