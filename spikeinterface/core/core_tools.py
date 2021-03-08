from pathlib import Path
import os
import numpy as np

from joblib import Parallel, delayed

def check_json(d):
    # quick hack to ensure json writable
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _check_json(v)
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
                    d[k] = [_check_json(v_el) for v_el in v]
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


def divide_recording_into_time_chunks(num_frames, chunk_size, padding_size):
    chunks = []
    ii = 0
    while ii < num_frames:
        ii2 = int(min(ii + chunk_size, num_frames))
        chunks.append(dict(
            istart=ii,
            iend=ii2,
            istart_with_padding=int(max(0, ii - padding_size)),
            iend_with_padding=int(min(num_frames, ii2 + padding_size))
        ))
        ii = ii2
    return chunks


def _write_dat_one_chunk(i, rec_arg, chunks, segment_index, rec_memmap, dtype, time_axis, verbose):
    chunk = chunks[i]

    if verbose:
        print(f"Writing chunk {i + 1} / {len(chunks)}")
    if isinstance(rec_arg, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(rec_arg)
    else:
        recording = rec_arg

    start_frame = chunk['istart']
    end_frame = chunk['iend']
    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    if time_axis == 1:
        traces = traces.T
    if dtype is not None:
        traces = traces.astype(dtype)

    rec_memmap[start_frame:end_frame, :] = traces


def write_binary_recording(recording, files_path=None, file_handle=None,
                               time_axis=0, dtype=None,
                               chunk_size=None, chunk_mb=500, n_jobs=1, joblib_backend='loky',
                               verbose=False):
    '''Saves the traces of a recording extractor in several binary .dat format.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be saved in .dat format
    file_path: str
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
    n_jobs: int
        Number of jobs to use (Default 1)
    joblib_backend: str
        Joblib backend for parallel processing ('loky', 'threading', 'multiprocessing')
    verbose: bool
        If True, output is verbose (when chunks are used)
    '''
    assert files_path is not None or file_handle is not None, "Provide 'file_path' or 'file handle'"
    
    # file path or file handle as list
    if files_path is None:
        assert file_handle is not None
        assert recording.get_num_segments() == 1, 'If file_handle is given then only deals with one segment'
        
    else:
        if not isinstance(files_path, list):
            files_path = [files_path]
        files_path = [Path(e) for e in files_path]
        files_path = [add_suffix(file_path, ['raw', 'bin', 'dat']) for file_path in files_path]

    if chunk_size is not None or chunk_mb is not None:
        if time_axis == 1:
            print("Chunking disabled due to 'time_axis' == 1")
            chunk_size = None
            chunk_mb = None

    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (recording.get_num_channels() * n_bytes)

    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    #~ elif n_jobs > 1:
        #~ if chunk_size is not None:
            #~ chunk_size /= n_jobs
    
    if not recording.is_dumpable:
        if n_jobs > 1:
            n_jobs = 1
            print("RecordingExtractor is not dumpable and can't be processed in parallel")
        rec_arg = recording
    else:
        if n_jobs > 1:
            rec_arg = recording.to_dict()
        else:
            rec_arg = recording

    if chunk_size is None:
        if file_handle is not None:
            # file handle is a special with only one file
            traces = recording.get_traces(segment_index=0)
            if time_axis == 1:
                traces = traces.T
                if dtype is not None:
                    traces = traces.astype(dtype)
            traces.tofile(file_handle)
        else:
            for segment_index in range(recording.get_num_segments()):
                with files_path[segment_index].open('wb') as f:
                    traces = recording.get_traces(segment_index=segment_index)
                    if time_axis == 1:
                        traces = traces.T
                    if dtype is not None:
                        traces = traces.astype(dtype)
                    traces.tofile(f)
    else:
        for segment_index in range(recording.get_num_segments()):
            # chunk size is not None
            num_frames = recording.get_num_samples(segment_index)
            num_channels = recording.get_num_channels()

            # chunk_size = num_bytes_per_chunk / num_bytes_per_frame
            chunks = divide_recording_into_time_chunks(
                num_frames=num_frames,
                chunk_size=chunk_size,
                padding_size=0
            )
            n_chunk = len(chunks)
            
            chunks_loop = range(n_chunk)
            if verbose and n_jobs == 1:
                chunks_loop = tqdm(chunks_loop, ascii=True, desc="Writing to binary .dat file")
            
            if file_handle is None:
                # file path case

                if time_axis == 0:
                    shape = (num_frames, num_channels)
                else:
                    shape = (num_channels, num_frames)

                file_path = files_path[segment_index]
                rec_memmap = np.memmap(str(file_path), dtype=dtype, mode='w+', shape=shape)
                
                if n_jobs == 1:
                    for i in chunks_loop:
                        _write_dat_one_chunk(i, rec_arg, chunks, segment_index, rec_memmap, dtype, time_axis, verbose=False)
                else:
                    Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                        delayed(_write_dat_one_chunk)(i, rec_arg, chunks, segment_index, rec_memmap, dtype, time_axis, verbose,)
                        for i in chunks_loop)
            
            else:
                # file handle case (one segment only)
                # Alessio : should be rewritten with memmap also because memmap accept filehandle
                for i in chunks_loop:
                    start_frame = chunks[i]['istart']
                    end_frame = chunks[i]['iend']
                    traces = recording.get_traces(segment_index=segment_index, 
                                start_frame=start_frame, end_frame=end_frame)
                    if time_axis == 1:
                        traces = traces.T
                    if dtype is not None:
                        traces = traces.astype(dtype)
                    file_handle.write(traces.tobytes())

    return file_path



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
    assert HAVE_H5, "To write to h5 you need to install h5py: pip install h5py"
    assert save_path is not None or file_handle is not None, "Provide 'save_path' or 'file handle'"

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix == '':
            # when suffix is already raw/bin/dat do not change it.
            save_path = save_path.parent / (save_path.name + '.h5')

    num_channels = recording.get_num_channels()
    num_frames = recording.get_num_frames()

    if file_handle is not None:
        assert isinstance(file_handle, h5py.File)
    else:
        file_handle = h5py.File(save_path, 'w')

    if dtype is None:
        dtype_file = recording.get_dtype()
    else:
        dtype_file = dtype

    if time_axis == 0:
        dset = file_handle.create_dataset(dataset_path, shape=(num_frames, num_channels), dtype=dtype_file)
    else:
        dset = file_handle.create_dataset(dataset_path, shape=(num_channels, num_frames), dtype=dtype_file)

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
        if time_axis == 0:
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
            chunk_frames = traces.shape[1]
            if dtype is not None:
                traces = traces.astype(dtype_file)
            if time_axis == 0:
                dset[chunk_start:chunk_start + chunk_frames] = traces.T
            else:
                dset[:, chunk_start:chunk_start + chunk_frames] = traces
            chunk_start += chunk_frames

    if save_path is not None:
        file_handle.close()
    return save_path


