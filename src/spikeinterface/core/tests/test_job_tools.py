import pytest
import os

import time

from spikeinterface.core import generate_recording, set_global_job_kwargs, get_global_job_kwargs, get_best_job_kwargs

from spikeinterface.core.job_tools import (
    divide_segment_into_chunks,
    ensure_n_jobs,
    ensure_chunk_size,
    ChunkRecordingExecutor,
    fix_job_kwargs,
    split_job_kwargs,
    divide_recording_into_chunks,
)


def test_divide_segment_into_chunks():
    chunks = divide_segment_into_chunks(10, 5)
    assert len(chunks) == 2
    chunks = divide_segment_into_chunks(11, 5)
    assert len(chunks) == 3
    assert chunks[0] == (0, 5)
    assert chunks[1] == (5, 10)
    assert chunks[2] == (10, 11)


def test_ensure_n_jobs():
    recording = generate_recording()

    n_jobs = ensure_n_jobs(recording)
    assert n_jobs == 1

    n_jobs = ensure_n_jobs(recording, n_jobs=0)
    assert n_jobs == 1

    n_jobs = ensure_n_jobs(recording, n_jobs=1)
    assert n_jobs == 1

    # check serializable
    n_jobs = ensure_n_jobs(recording.save(), n_jobs=-1)
    assert n_jobs > 1


def test_ensure_chunk_size():
    recording = generate_recording(num_channels=2, durations=[5.0, 2.5])  # This is the default value for two semgents
    dtype = recording.get_dtype()
    assert dtype == "float32"

    chunk_size = ensure_chunk_size(recording, total_memory="512M", chunk_size=None, chunk_memory=None, n_jobs=2)
    assert chunk_size == 32000000

    chunk_size = ensure_chunk_size(recording, chunk_memory="256M")
    assert chunk_size == 32000000

    chunk_size = ensure_chunk_size(recording, chunk_memory="1k")
    assert chunk_size == 125

    chunk_size = ensure_chunk_size(recording, chunk_memory="1G")
    assert chunk_size == 125000000

    chunk_size = ensure_chunk_size(recording, chunk_duration=1.5)
    assert chunk_size == 45000

    chunk_size = ensure_chunk_size(recording, chunk_duration="1.5s")
    assert chunk_size == 45000

    chunk_size = ensure_chunk_size(recording, chunk_duration="500ms")
    assert chunk_size == 15000

    # Test edge case to define single chunk for n_jobs=1
    chunk_size = ensure_chunk_size(recording, n_jobs=1, chunk_size=None)
    chunks = divide_recording_into_chunks(recording, chunk_size)
    assert len(chunks) == recording.get_num_segments()
    for chunk in chunks:
        segment_index, start_frame, end_frame = chunk
        assert start_frame == 0
        assert end_frame == recording.get_num_frames(segment_index=segment_index)


def func(segment_index, start_frame, end_frame, worker_dict):
    import os

    #  print('func', segment_index, start_frame, end_frame, worker_dict, os.getpid())
    time.sleep(0.010)
    # time.sleep(1.0)
    return os.getpid()


def init_func(arg1, arg2, arg3):
    worker_dict = {}
    worker_dict["arg1"] = arg1
    worker_dict["arg2"] = arg2
    worker_dict["arg3"] = arg3
    return worker_dict


def test_ChunkRecordingExecutor():
    recording = generate_recording(num_channels=2)

    init_args = "a", 120, "yep"

    # no chunk
    processor = ChunkRecordingExecutor(
        recording, func, init_func, init_args, verbose=True, progress_bar=False, n_jobs=1, chunk_size=None
    )
    processor.run()

    # simple gathering function
    def gathering_result(res):
        # print(res)
        pass

    # chunk + loop + gather_func
    processor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        verbose=True,
        progress_bar=False,
        gather_func=gathering_result,
        n_jobs=1,
        chunk_memory="500k",
    )
    processor.run()

    # more adavnce trick : gathering using class with callable
    class GatherClass:
        def __init__(self):
            self.pos = 0

        def __call__(self, res):
            self.pos += 1
            # print(self.pos, res)
            pass

    gathering_func2 = GatherClass()

    # process + gather_func
    processor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        verbose=True,
        progress_bar=True,
        gather_func=gathering_func2,
        pool_engine="process",
        n_jobs=2,
        chunk_duration="200ms",
        job_name="job_name",
    )
    processor.run()
    num_chunks = len(divide_recording_into_chunks(recording, processor.chunk_size))

    assert gathering_func2.pos == num_chunks

    # process spawn
    processor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        verbose=True,
        progress_bar=True,
        pool_engine="process",
        mp_context="spawn",
        n_jobs=2,
        chunk_duration="200ms",
        job_name="job_name",
    )
    processor.run()

    # thread
    processor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        verbose=True,
        progress_bar=True,
        pool_engine="thread",
        n_jobs=2,
        chunk_duration="200ms",
        job_name="job_name",
    )
    processor.run()


def test_fix_job_kwargs():
    # test negative n_jobs
    job_kwargs = dict(n_jobs=-1, progress_bar=False, chunk_duration="1s")
    fixed_job_kwargs = fix_job_kwargs(job_kwargs)
    assert fixed_job_kwargs["n_jobs"] == os.cpu_count()

    # test float n_jobs
    job_kwargs = dict(n_jobs=0.5, progress_bar=False, chunk_duration="1s")
    fixed_job_kwargs = fix_job_kwargs(job_kwargs)
    if int(0.5 * os.cpu_count()) > 1:
        assert fixed_job_kwargs["n_jobs"] == int(0.5 * os.cpu_count())
    else:
        assert fixed_job_kwargs["n_jobs"] == 1

    # test float value > 1 is cast to correct int
    job_kwargs = dict(n_jobs=float(os.cpu_count()), progress_bar=False, chunk_duration="1s")
    fixed_job_kwargs = fix_job_kwargs(job_kwargs)
    assert fixed_job_kwargs["n_jobs"] == os.cpu_count()

    # test wrong keys
    with pytest.raises(AssertionError):
        job_kwargs = dict(n_jobs=0, progress_bar=False, chunk_duration="1s", other_param="other")
        fixed_job_kwargs = fix_job_kwargs(job_kwargs)

    # test mutually exclusive
    _old_global = get_global_job_kwargs().copy()
    set_global_job_kwargs(chunk_memory="50M")
    job_kwargs = dict()
    fixed_job_kwargs = fixed_job_kwargs = fix_job_kwargs(job_kwargs)
    assert "chunk_memory" in fixed_job_kwargs

    job_kwargs = dict(chunk_duration="300ms")
    fixed_job_kwargs = fixed_job_kwargs = fix_job_kwargs(job_kwargs)
    assert "chunk_memory" not in fixed_job_kwargs
    assert fixed_job_kwargs["chunk_duration"] == "300ms"
    set_global_job_kwargs(**_old_global)


def test_split_job_kwargs():
    kwargs = dict(n_jobs=2, progress_bar=False, other_param="other")
    specific_kwargs, job_kwargs = split_job_kwargs(kwargs)
    assert (
        "other_param" in specific_kwargs and "n_jobs" not in specific_kwargs and "progress_bar" not in specific_kwargs
    )
    assert "other_param" not in job_kwargs and "n_jobs" in job_kwargs and "progress_bar" in job_kwargs


def func2(segment_index, start_frame, end_frame, worker_dict):
    time.sleep(0.010)
    # print(os.getpid(), worker_dict["worker_index"])
    return worker_dict["worker_index"]


def init_func2(worker_index):
    # this leave time for other thread/process to start
    # print('in init_func2 with worker_index', worker_index)
    time.sleep(0.010)
    worker_dict = {}
    return worker_dict


def test_worker_index():
    recording = generate_recording(num_channels=2)
    init_args = tuple()

    for i in range(2):
        # making this 2 times ensure to test that global variables are correctly reset
        for pool_engine in ("process", "thread"):
            # print(pool_engine)
            processor = ChunkRecordingExecutor(
                recording,
                func2,
                init_func2,
                init_args,
                progress_bar=False,
                gather_func=None,
                pool_engine=pool_engine,
                n_jobs=2,
                handle_returns=True,
                chunk_duration="200ms",
                need_worker_index=True,
            )
            res = processor.run()
            # we should have a mix of 0 and 1
            assert 0 in res
            assert 1 in res


def test_get_best_job_kwargs():
    job_kwargs = get_best_job_kwargs()
    print(job_kwargs)


# def quick_becnhmark():
#     # keep this commented do not remove

#     from spikeinterface.generation import generate_drifting_recording
#     from spikeinterface.sortingcomponents.peak_detection import detect_peaks
#     from spikeinterface import get_noise_levels
#     import time

#     all_job_kwargs = [
#         dict(pool_engine="process", n_jobs=2, mp_context="spawn", max_threads_per_worker=2),
#         dict(pool_engine="process", n_jobs=4, mp_context="spawn", max_threads_per_worker=1),
#         dict(pool_engine="thread", n_jobs=4, mp_context=None, max_threads_per_worker=1),
#         dict(pool_engine="thread", n_jobs=2, mp_context=None, max_threads_per_worker=2),
#         dict(n_jobs=1),
#     ]


#     rec, _, sorting = generate_drifting_recording(
#         num_units=50,
#         duration=120.0,
#         sampling_frequency=30000.0,
#         probe_name="Neuropixels-128",

#     )
#     # print(rec)

#     noise_levels = get_noise_levels(rec, return_in_uV=False)
#     for job_kwargs in all_job_kwargs:
#         print()
#         print(job_kwargs)
#         t0 = time.perf_counter()
#         peaks = detect_peaks(rec, method="locally_exclusive", noise_levels=noise_levels, **job_kwargs)
#         t1 = time.perf_counter()
#         print("time included the spawn:", t1-t0)


if __name__ == "__main__":
    # test_divide_segment_into_chunks()
    # test_ensure_n_jobs()
    # test_ensure_chunk_size()
    # test_ChunkRecordingExecutor()
    # test_fix_job_kwargs()
    # test_split_job_kwargs()
    test_worker_index()
    # test_get_best_job_kwargs()

    # quick_becnhmark()
