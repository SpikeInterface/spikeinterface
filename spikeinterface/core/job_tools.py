"""
Some utils to handle parallel jobs on top of job and/or loky
"""
from pathlib import Path
import numpy as np
import platform
import os

import joblib
import sys
import contextlib
from tqdm.auto import tqdm

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from threadpoolctl import threadpool_limits


_shared_job_kwargs_doc = """**job_kwargs: keyword arguments for parallel processing:
            * chunk_duration or chunk_size or chunk_memory or total_memory
                - chunk_size: int
                    Number of samples per chunk
                - chunk_memory: str
                    Memory usage for each job (e.g. '100M', '1G')
                - total_memory: str
                    Total memory usage (e.g. '500M', '2G')
                - chunk_duration : str or float or None
                    Chunk duration in s if float or with units if str (e.g. '1s', '500ms')
            * n_jobs: int
                Number of jobs to use. With -1 the number of jobs is the same as number of cores
            * progress_bar: bool
                If True, a progress bar is printed
            * mp_context: str or None
                Context for multiprocessing. It can be None (default), "fork" or "spawn". 
                Note that "fork" is only available on UNIX systems
    """
    
job_keys = ('n_jobs', 'total_memory', 'chunk_size', 'chunk_memory', 'chunk_duration', 'progress_bar', 
            'mp_context', 'verbose', 'max_threads_per_process')


def fix_job_kwargs(runtime_job_kwargs):
    from .globals import get_global_job_kwargs
    job_kwargs = get_global_job_kwargs()
    
    for k in runtime_job_kwargs:
        assert k in job_keys, (f"{k} is not a valid job keyword argument. "
                               f"Available keyword arguments are: {list(job_keys)}")
    job_kwargs.update(runtime_job_kwargs)

    # if n_jobs is -1, set to os.cpu_count()
    if "n_jobs" in job_kwargs:
        assert isinstance(job_kwargs["n_jobs"], (float, np.integer, int))
        if isinstance(job_kwargs["n_jobs"], float):
            n_jobs = int(job_kwargs["n_jobs"] * os.cpu_count())
        elif job_kwargs["n_jobs"] < 0:            
            n_jobs = os.cpu_count() + 1 + job_kwargs["n_jobs"]
        else:
            n_jobs = job_kwargs["n_jobs"]
        job_kwargs["n_jobs"] = max(n_jobs, 1)
    return job_kwargs


def split_job_kwargs(mixed_kwargs):
    """
    This function splits mixed kwargs into job_kwargs and specific_kwargs.
    This can be useful for some function with generic signature
    mixing specific and job kwargs.
    """
    job_kwargs = {}
    specific_kwargs = {}
    for k, v in mixed_kwargs.items():
        if k in job_keys:
            job_kwargs[k] = v
        else:
            specific_kwargs[k] = v
    job_kwargs = fix_job_kwargs(job_kwargs)
    return specific_kwargs, job_kwargs


# from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def divide_segment_into_chunks(num_frames, chunk_size):
    if chunk_size is None:
        chunks = [(0, num_frames)]
    else:
        n = num_frames // chunk_size

        frame_starts = np.arange(n) * chunk_size
        frame_stops = frame_starts + chunk_size

        frame_starts = frame_starts.tolist()
        frame_stops = frame_stops.tolist()

        if (num_frames % chunk_size) > 0:
            frame_starts.append(n * chunk_size)
            frame_stops.append(num_frames)

        chunks = list(zip(frame_starts, frame_stops))

    return chunks


def divide_recording_into_chunks(recording, chunk_size):
    all_chunks = []
    for segment_index in range(recording.get_num_segments()):
        num_frames = recording.get_num_samples(segment_index)
        chunks = divide_segment_into_chunks(num_frames, chunk_size)
        all_chunks.extend([(segment_index, frame_start, frame_stop) for frame_start, frame_stop in chunks])
    return all_chunks


_exponents = {'k': 1e3, 'M': 1e6, 'G': 1e9}


def _mem_to_int(mem):
    suffix = mem[-1]
    assert suffix in _exponents
    mem = int(float(mem[:-1]) * _exponents[suffix])
    return mem


def ensure_n_jobs(recording, n_jobs=1):
    if n_jobs == -1:
        n_jobs = joblib.cpu_count()
    elif n_jobs == 0:
        n_jobs = 1
    elif n_jobs is None:
        n_jobs = 1

    version = sys.version_info

    if (n_jobs != 1) and not (version.major >= 3 and version.minor >= 7):
        print(f"Python {sys.version} does not support parallel processing")
        n_jobs = 1

    if not recording.check_if_dumpable():
        if n_jobs != 1:
            raise RuntimeError(
                "Recording is not dumpable and can't be processed in parallel. "
                "You can use the `recording.save()` function to make it dumpable or set 'n_jobs' to 1."
            )

    return n_jobs


def ensure_chunk_size(recording, total_memory=None, chunk_size=None, chunk_memory=None, chunk_duration=None, n_jobs=1, 
                      **other_kwargs):
    """
    'chunk_size' is the traces.shape[0] for each worker.

    Flexible chunk_size setter with 3 ways:
        * "chunk_size": is the length in sample for each chunk independently of channel count and dtype.
        * "chunk_memory": total memory per chunk per worker
        * "total_memory": total memory over all workers.

    If chunk_size/chunk_memory/total_memory are all None then there is no chunk computing
    and the full trace is retrieved at once.

    Parameters
    ----------
    chunk_size: int or None
        size for one chunk per job
    chunk_memory: str or None
        must endswith 'k', 'M' or 'G'
    total_memory: str or None
        must endswith 'k', 'M' or 'G'
    chunk_duration: None or float or str
        Units are second if float.
        If str then the str must contain units(e.g. '1s', '500ms')
    """
    if chunk_size is not None:
        # manual setting
        chunk_size = int(chunk_size)
    elif chunk_memory is not None:
        assert total_memory is None
        # set by memory per worker size
        chunk_memory = _mem_to_int(chunk_memory)
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        num_channels = recording.get_num_channels()
        chunk_size = int(chunk_memory / (num_channels * n_bytes))
    elif total_memory is not None:
        # clip by total memory size
        n_jobs = ensure_n_jobs(recording, n_jobs=n_jobs)
        total_memory = _mem_to_int(total_memory)
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        num_channels = recording.get_num_channels()
        chunk_size = int(total_memory / (num_channels * n_bytes * n_jobs))
    elif chunk_duration is not None:
        if isinstance(chunk_duration, float):
            chunk_size = int(chunk_duration * recording.get_sampling_frequency())
        elif isinstance(chunk_duration, str):
            if chunk_duration.endswith('ms'):
                chunk_duration = float(chunk_duration.replace('ms', '')) / 1000.
            elif chunk_duration.endswith('s'):
                chunk_duration = float(chunk_duration.replace('s', ''))
            else:
                raise ValueError('chunk_duration must ends with s or ms') 
            chunk_size = int(chunk_duration * recording.get_sampling_frequency())
        else:
            raise ValueError('chunk_duration must be str or float') 
    else:
        if n_jobs == 1:
            # not chunk computing
            chunk_size = None
        else:
            raise ValueError('For n_jobs >1 you must specify total_memory or chunk_size or chunk_memory')

    return chunk_size


class ChunkRecordingExecutor:
    """
    Core class for parallel processing to run a "function" over chunks on a recording.

    It supports running a function:
        * in loop with chunk processing (low RAM usage)
        * at once if chunk_size is None (high RAM usage)
        * in parallel with ProcessPoolExecutor (higher speed)

    The initializer ('init_func') allows to set a global context to avoid heavy serialization
    (for examples, see implementation in `core.WaveformExtractor`).

    Parameters
    ----------
    recording: RecordingExtractor
        The recording to be processed
    func: function
        Function that runs on each chunk
    init_func: function
        Initializer function to set the global context (accessible by 'func')
    init_args: tuple
        Arguments for init_func
    verbose: bool
        If True, output is verbose
    progress_bar: bool
        If True, a progress bar is printed to monitor the progress of the process
    handle_returns: bool
        If True, the function can return values
    n_jobs: int
        Number of jobs to be used (default 1). Use -1 to use as many jobs as number of cores
    total_memory: str
        Total memory (RAM) to use (e.g. "1G", "500M")
    chunk_memory: str
        Memory per chunk (RAM) to use (e.g. "1G", "500M")
    chunk_size: int or None
        Size of each chunk in number of samples. If 'total_memory' or 'chunk_memory' are used, it is ignored.
    chunk_duration : str or float or None
        Chunk duration in s if float or with units if str (e.g. '1s', '500ms')
    mp_context : str or None
        "fork" (default) or "spawn". If None, the context is taken by the recording.get_preferred_mp_context().
        "fork" is only available on UNIX systems.
    job_name: str
        Job name
    max_threads_per_process: int or None
        Limit the number of thread per process using threadpoolctl modules.
        This used only when n_jobs>1
        If None, no limits.

    Returns
    -------
    res: list
        If 'handle_returns' is True, the results for each chunk process
    """

    def __init__(self, recording, func, init_func, init_args, verbose=False, progress_bar=False, handle_returns=False,
                 n_jobs=1, total_memory=None, chunk_size=None, chunk_memory=None, chunk_duration=None,
                 mp_context=None, job_name='', max_threads_per_process=1):
        self.recording = recording
        self.func = func
        self.init_func = init_func
        self.init_args = init_args
        
        if mp_context is None:
            mp_context = recording.get_preferred_mp_context()
        if mp_context is not None and platform.system() == "Windows":
            assert mp_context != "fork", "'fork' mp_context not supported on Windows!"
                
        self.mp_context = mp_context

        self.verbose = verbose
        self.progress_bar = progress_bar

        self.handle_returns = handle_returns

        self.n_jobs = ensure_n_jobs(recording, n_jobs=n_jobs)
        self.chunk_size = ensure_chunk_size(recording,
                                            total_memory=total_memory, chunk_size=chunk_size,
                                            chunk_memory=chunk_memory, chunk_duration=chunk_duration,
                                            n_jobs=self.n_jobs)
        self.job_name = job_name
        self.max_threads_per_process = max_threads_per_process
        
        if verbose:
            print(self.job_name, 'with n_jobs =', self.n_jobs, 'and chunk_size =', self.chunk_size)

    def run(self):
        """
        Runs the defined jobs.
        """
        all_chunks = divide_recording_into_chunks(self.recording, self.chunk_size)

        if self.handle_returns:
            returns = []
        else:
            returns = None

        import sys
        if self.n_jobs != 1 and not (sys.version_info >= (3, 8)):
            self.n_jobs = 1

        if self.n_jobs == 1:
            if self.progress_bar:
                all_chunks = tqdm(all_chunks, ascii=True, desc=self.job_name)

            worker_ctx = self.init_func(*self.init_args)
            for segment_index, frame_start, frame_stop in all_chunks:
                res = self.func(segment_index, frame_start, frame_stop, worker_ctx)
                if self.handle_returns:
                    returns.append(res)
        else:
            n_jobs = min(self.n_jobs, len(all_chunks))
            ######## Do you want to limit the number of threads per process?
            ######## It has to be done to speed up numpy a lot if multicores
            ######## Otherwise, np.dot will be slow. How to do that, up to you
            ######## This is just a suggestion, but here it adds a dependency

            # parallel
            with ProcessPoolExecutor(max_workers=n_jobs,
                                     initializer=worker_initializer,
                                     mp_context=mp.get_context(self.mp_context),
                                     initargs=(self.func, self.init_func, self.init_args, self.max_threads_per_process)) as executor:

                results = executor.map(function_wrapper, all_chunks)

                if self.progress_bar:
                    results = tqdm(results, desc=self.job_name, total=len(all_chunks))

                if self.handle_returns:
                    for res in results:
                        returns.append(res)
                else:
                    for res in results:
                        pass

        return returns


# see
# https://stackoverflow.com/questions/10117073/how-to-use-initializer-to-set-up-my-multiprocess-pool
# the tricks is : theses 2 variables are global per worker
# so they are not share in the same process
global _worker_ctx
global _func


def worker_initializer(func, init_func, init_args, max_threads_per_process):
    global _worker_ctx
    if max_threads_per_process is None:
        _worker_ctx = init_func(*init_args)
    else:
        with threadpool_limits(limits=max_threads_per_process):
            _worker_ctx = init_func(*init_args)
    _worker_ctx['max_threads_per_process'] = max_threads_per_process
    global _func
    _func = func


def function_wrapper(args):
    segment_index, start_frame, end_frame = args
    global _func
    global _worker_ctx
    max_threads_per_process = _worker_ctx['max_threads_per_process']
    if max_threads_per_process is None:
        return _func(segment_index, start_frame, end_frame, _worker_ctx)
    else:
        with threadpool_limits(limits=max_threads_per_process):
            return _func(segment_index, start_frame, end_frame, _worker_ctx)
