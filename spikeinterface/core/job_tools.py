"""
Some utils to handle parral jobs on top of job and/or loky

"""
from pathlib import Path
import numpy as np

from joblib import Parallel, delayed



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
    

_exponents = {'k': 1e3, 'M':1e6, 'G':1e9}
def ensure_chunk_size(chunk_size):
    """
    Parameters
    ----------
    chunk_size: int or str or flost
        Ensure chunk_size is integer.
        For str chunk_size can "100M" "0.5G" "500k"
    
    
    
    """
    if isinstance(chunk_size, int):
        return chunk_size
    elif isinstance(chunk_size, float):
        return int(chunk_size)
    elif isinstance(chunk_size, str):
        suffix = chunk_size[-1]
        assert suffix in _exponents
        chunk_size = int(float(chunk_size[:-1]) * _exponents[suffix])
        return chunk_size
    else:
        raise ValueError(f'Wrong chunk_size {chunk_size}') 
