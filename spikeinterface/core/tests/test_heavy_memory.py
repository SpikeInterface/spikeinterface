import pytest 
from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN

import psutil

from spikeinterface.core.generate import generate_lazy_random_recording


def test_writing_overflow():
    
    # Convert the total memory to GB
    total_memory = psutil.virtual_memory().total
    total_memory_GiB = total_memory / (1024 * 1024 * 1024)

    # Get the model name of the processor
    processor_name = psutil.cpu_info().current_processor

    # Print the RAM and processor information
    print(f"{total_memory_GiB=}")
    print(f"{processor_name=}")
    print("=================")
    print("Running tests")
    print("=================")
    
    full_traces_size_GiB = 8.0
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

def test_writing_overflow_failing_non_parallel():
    # Convert the total memory to GB
    total_memory = psutil.virtual_memory().total
    total_memory_GiB = total_memory / (1024 * 1024 * 1024)

    # Get the model name of the processor
    processor_name = psutil.cpu_info().current_processor

    # Print the RAM and processor information
    print(f"{total_memory_GiB=}")
    print(f"{processor_name=}")
    print("=================")
    print("Running tests")
    print("=================")
    
    full_traces_size_GiB = 8.0
    job_kwargs = dict(n_jobs=1, total_memory="8G", verbose=True, progress_bar=True)
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
