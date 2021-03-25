"""
Some utils to handle parral jobs on top of job and/or loky

"""
from pathlib import Path
import numpy as np

from joblib import Parallel, delayed
import joblib

import loky

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except:
    HAVE_TQDM = False


        

def devide_into_chunks(num_frames, chunk_size):
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
    
    

#~ def divide_recording_into_time_chunks(num_frames, chunk_size, padding_size):
    #~ chunks = []
    #~ ii = 0
    #~ while ii < num_frames:
        #~ ii2 = int(min(ii + chunk_size, num_frames))
        #~ chunks.append(dict(
            #~ istart=ii,
            #~ iend=ii2,
            #~ istart_with_padding=int(max(0, ii - padding_size)),
            #~ iend_with_padding=int(min(num_frames, ii2 + padding_size))
        #~ ))
        #~ ii = ii2
    #~ return chunks


_exponents = {'k': 1e3, 'M':1e6, 'G':1e9}

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

    if not recording.is_dumpable:
        if n_jobs > 1:
            n_jobs = 1
            print("RecordingExtractor is not dumpable and can't be processed in parallel")
    
    return n_jobs

    

def ensure_chunk_size(recording, total_memory=None, chunk_size=None, chunk_memory=None, n_jobs=1, **other_kwargs):
    """
    'chunk_size' is the traces.shape[0] for each worker.
    
    Flexible chunk_size setter with 3 ways:
    
        * "chunk_size": is the length in sample for each
                chunk independantly of channel count and dtype items size.
        * "chunk_memory": total memory per chunk per worker
        * "total_memory": total memory over all workers.
    
    If chunk_size/chunk_memory/total_memory are all None then chunk_size is None and
    means that there is no chunk computing the full trace is retrieve once.
    
    Parameters
    ----------
    chunk_size: int or None
        size for one chunk per job
    chunk_memory: str or None
        must endswith 'k', 'M' or 'G'
    total_memory: str or None
        must endswith 'k', 'M' or 'G'
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
    else:
        if n_jobs == 1:
            # not chunk computing
            chunk_size = None
        else:
            raise ValueError('For n_jobs >1 you must specify total_memory or chunk_size or chunk_memory')
    
    return chunk_size




class ChunkRecordingProcessor:
    """
    Helper class that runner a "function" over chunk on a recording
    
    Do it in loop or in parrael witj joblib/loky or at once if chunk_size is None.
    
    Handle initializer when needed to avoid heavy serilization of args.
    """
    
    def __init__(self, recording, func, init_func, init_args, verbose=False, progress_bar=False,
                n_jobs=1,  total_memory=None, chunk_size=None, chunk_memory=None,
                job_name=''):
        
        self.recording = recording
        self.func = func
        self.init_func = init_func
        self.init_args = init_args

        self.verbose = verbose
        self.progress_bar =progress_bar
        
        self.n_jobs = ensure_n_jobs(recording, n_jobs=n_jobs)
        self.chunk_size = ensure_chunk_size(recording,
                total_memory=total_memory, chunk_size=chunk_size, 
                chunk_memory=chunk_memory, n_jobs=self.n_jobs)
        self.job_name = job_name
        
        if verbose:
            print(self.job_name, 'with',  'n_jobs', self.n_jobs, ' chunk_size',  self.chunk_size)
        
    
    def run(self):
        if self.n_jobs == 1:
            local_dict = self.init_func(*self.init_args)
            for segment_index in range(self.recording.get_num_segments()):
                num_frames = self.recording.get_num_samples(segment_index)
                chunks = devide_into_chunks(num_frames, self.chunk_size)
                
                if self.progress_bar and HAVE_TQDM:
                    chunks = tqdm(chunks, ascii=True, desc=self.job_name + f'segment {segment_index}')

                for frame_start, frame_stop in chunks:
                    self.func(segment_index, frame_start, frame_stop, local_dict)

        else:
            all_chunks = []
            for segment_index in range(self.recording.get_num_segments()):
                num_frames = self.recording.get_num_samples(segment_index)
                chunks = devide_into_chunks(num_frames, self.chunk_size)
                all_chunks.extend([(segment_index,  frame_start, frame_stop) for  frame_start, frame_stop in chunks])

            # if self.verbose:
            #   print('num chunks to compute', len(all_chunks))

            # parallel
            # we force reuse to false because it lead to bugs....
            executor = loky.get_reusable_executor(max_workers=self.n_jobs,
                    initializer=worker_initializer,
                    initargs=(self.func, self.init_func, self.init_args),
                    context="loky", timeout=10.,
                    reuse=False,
                    kill_workers=True)
            
            results = executor.map(function_wrapper, all_chunks)
            
            if self.progress_bar and HAVE_TQDM:
                results = tqdm(results)
            
            for res in results:
                pass

# see
# https://stackoverflow.com/questions/10117073/how-to-use-initializer-to-set-up-my-multiprocess-pool 
# theses 2 global are global per worker
# so they are not share
global local_dict
global _func
def worker_initializer(func, init_func, init_args):
    global local_dict
    local_dict = init_func(*init_args)
    global _func
    _func = func

def function_wrapper(args):
    segment_index,start_frame, end_frame = args
    global _func
    global local_dict
    _func(segment_index,start_frame, end_frame, local_dict)

