import pytest 
from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN

from spikeinterface.core.generate import generate_lazy_random_recording


def test_writing_overflow():
    
    full_traces_size_GiB = 15.0  # 15 GiB
    job_kwargs = dict(n_jobs=-1, total_memory="1G", verbose=True, progress_bar=True)
    # job_kwargs = dict(n_jobs=-1, chunk_memory="1G", verbose=True, progress_bar=True)
    large_recording = generate_lazy_random_recording(full_traces_size_GiB=full_traces_size_GiB)
    
    binary_recoder_cache = large_recording.save_to_folder(**job_kwargs)
    print(binary_recoder_cache)

    print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_SELF).ru_ixrss / (1024 * 1024))
    print("children")
    print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024))
    print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_ixrss / (1024 * 1024))

if __name__ == '__main__':
    test_writing_overflow()
