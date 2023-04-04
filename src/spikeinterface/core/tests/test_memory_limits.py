import pytest 
import platform

import psutil

from spikeinterface.core.generate import generate_lazy_recording


# def test_writing_overflow():
#     platform_system = platform.system()
#     partition_path = "C:" if platform_system == "Windows" else "/" 
#     hdd = psutil.disk_usage(partition_path)

#     print(f"Total: {hdd.total / (2**30)} GiB" )
#     print(f"Used: {hdd.used / (2**30)} GiB")
#     print(f"Free: {hdd.free / (2**30)} GiB")

#     # Convert the total memory to GB
#     total_memory = psutil.virtual_memory().total
#     total_memory_GiB_total = total_memory / (1024 * 1024 * 1024)

#     total_memory = psutil.virtual_memory().available
#     total_memory_GiB_availalble = total_memory / (1024 * 1024 * 1024)
#     # Get the model name of the processor

#     # Print the RAM and processor information
#     print(f"{total_memory_GiB_total=}")
#     print(f"{total_memory_GiB_availalble=}")

#     print("=================")
#     print("Running tests")
#     print("=================")

#     full_traces_size_GiB = 10.0  #  Both linux and windows have 7 GiB of ram available
#     job_kwargs = dict(n_jobs=-1, total_memory="1G", verbose=True, progress_bar=True)
#     # job_kwargs = dict(n_jobs=-1, chunk_memory="1G", verbose=True, progress_bar=True)
#     large_recording = generate_lazy_recording(full_traces_size_GiB=full_traces_size_GiB)

#     binary_recoder_cache = large_recording.save_to_folder(**job_kwargs)
#     print(binary_recoder_cache)

#     print("=================")
#     print("Post tests memory")
#     print("=================")

#     # Convert the total memory to GB
#     total_memory = psutil.virtual_memory().total
#     total_memory_GiB_total = total_memory / (1024 * 1024 * 1024)

#     total_memory = psutil.virtual_memory().available
#     total_memory_GiB_availalble = total_memory / (1024 * 1024 * 1024)
#     # Get the model name of the processor

#     # Print the RAM and processor information
#     print(f"{total_memory_GiB_total=}")
#     print(f"{total_memory_GiB_availalble=}")


#     if platform_system == "Linux":
#         from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN
#         print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_SELF).ru_maxrss / (1024 * 1024))
#         print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_SELF).ru_ixrss / (1024 * 1024))
#         print("children")
#         print("Peak resident memory ru_maxrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024))
#         print("Peak virtual memory ru_ru_ixrss (GiB):", getrusage(RUSAGE_CHILDREN).ru_ixrss / (1024 * 1024))

def test_reading_from_memamp_overflow(tmp_path):
    platform_system = platform.system()
    partition_path = "C:" if platform_system == "Windows" else "/" 
    hdd = psutil.disk_usage(partition_path)


    available_hard_drive_GiB = hdd.free / (2**30)
    total_memory = psutil.virtual_memory().total
    total_memory_available_GiB = total_memory / (1024 * 1024 * 1024)

    # Convert the total memory to GB
    full_traces_size_GiB = 10.0  #  Both linux and windows have 7 GiB of ram available

    large_recording = generate_lazy_recording(full_traces_size_GiB=full_traces_size_GiB)
    folder = "/tmp/this_terrible_test/"
    folder = tmp_path / "this_terrible_test"
    verbose = True
    job_kwargs = dict(total_memory="1G", n_jobs=2, verbose=verbose, progress_bar=True)
    large_recording.save_to_folder(folder=folder, **job_kwargs)

    from spikeinterface.core.base import load_extractor
    recording = load_extractor(folder)

    def  _dummy_chunk_operation(segment_index, start_frame, end_frame, worker_ctx):
        # recover variables of the worker
        recording = worker_ctx["recording"]

        # apply function
        traces = recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            segment_index=segment_index,
        )
        traces[:]
        
    def _init_dummy_worker(recording):
        # create a local dict per worker
        worker_ctx = {}
        worker_ctx["recording"] = recording

        return worker_ctx

    func = _dummy_chunk_operation
    init_func = _init_dummy_worker
    init_args = (recording, )
    from spikeinterface.core.job_tools import ChunkRecordingExecutor
    executor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        **job_kwargs,
    )
    executor.run()
if __name__ == '__main__':
    test_writing_overflow()