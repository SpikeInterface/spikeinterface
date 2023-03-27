import pytest 
import platform

import psutil

from spikeinterface.core.generate import generate_lazy_random_recording


def test_writing_overflow():
    platform_system = platform.system()
    partition_path = "C:" if platform_system == "Windows" else "/" 
    hdd = psutil.disk_usage(partition_path)

    print(f"Total: {hdd.total / (2**30)} GiB" )
    print(f"Used: {hdd.used / (2**30)} GiB")
    print(f"Free: {hdd.free / (2**30)} GiB")

    # Convert the total memory to GB
    total_memory = psutil.virtual_memory().total
    total_memory_GiB_total = total_memory / (1024 * 1024 * 1024)

    total_memory = psutil.virtual_memory().available
    total_memory_GiB_availalble = total_memory / (1024 * 1024 * 1024)
    # Get the model name of the processor

    # Print the RAM and processor information
    print(f"{total_memory_GiB_total=}")
    print(f"{total_memory_GiB_availalble=}")

    print("=================")
    print("Running tests")
    print("=================")

    full_traces_size_GiB = 10.0  #  Both linux and windows have 7 GiB of ram available
    job_kwargs = dict(n_jobs=-1, total_memory="1G", verbose=True, progress_bar=True)
    # job_kwargs = dict(n_jobs=-1, chunk_memory="1G", verbose=True, progress_bar=True)
    large_recording = generate_lazy_random_recording(full_traces_size_GiB=full_traces_size_GiB)

    binary_recoder_cache = large_recording.save_to_folder(**job_kwargs)
    print(binary_recoder_cache)

    print("=================")
    print("Post tests memory")
    print("=================")

    # Convert the total memory to GB
    total_memory = psutil.virtual_memory().total
    total_memory_GiB_total = total_memory / (1024 * 1024 * 1024)

    total_memory = psutil.virtual_memory().available
    total_memory_GiB_availalble = total_memory / (1024 * 1024 * 1024)
    # Get the model name of the processor

    # Print the RAM and processor information
    print(f"{total_memory_GiB_total=}")
    print(f"{total_memory_GiB_availalble=}")


    if platform_system == "Linux":
        from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN
        print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024))
        print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_SELF).ru_ixrss / (1024 * 1024))
        print("children")
        print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024))
        print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_ixrss / (1024 * 1024))

def test_writing_overflow_failing_non_parallel():

    
    # Convert the total memory to GB
    total_memory = psutil.virtual_memory().total
    total_memory_GiB_total = total_memory / (1024 * 1024 * 1024)

    total_memory = psutil.virtual_memory().available
    total_memory_GiB_availalble = total_memory / (1024 * 1024 * 1024)
    # Get the model name of the processor

    # Print the RAM and processor information
    print(f"{total_memory_GiB_total=}")
    print(f"{total_memory_GiB_availalble=}")

    print("=================")
    print("Running tests")
    print("=================")

    full_traces_size_GiB = 8.0
    job_kwargs = dict(n_jobs=1, total_memory="8G", verbose=True, progress_bar=True)
    # job_kwargs = dict(n_jobs=-1, chunk_memory="1G", verbose=True, progress_bar=True)
    large_recording = generate_lazy_random_recording(full_traces_size_GiB=full_traces_size_GiB)

    binary_recoder_cache = large_recording.save_to_folder(**job_kwargs)
    print(binary_recoder_cache)


    platform_system = platform.system()
    if platform_system == "Linux":
        from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN
        print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024))
        print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_SELF).ru_ixrss / (1024 * 1024))
        print("children")
        print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024))
        print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_ixrss / (1024 * 1024))


if __name__ == '__main__':
    test_writing_overflow()