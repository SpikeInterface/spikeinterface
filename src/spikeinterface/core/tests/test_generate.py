import pytest
import psutil

import numpy as np

from spikeinterface.core.generate import GeneratorRecording, generate_lazy_recording

mode_list = ["deterministic", "pure_noise"]


def measure_memory_allocation(measure_in_process: bool = True) -> float:
    """
    A local utility to measure memory allocation at a specific point in time.
    Can measure either the process resident memory or system wide memory available

    Uses psutil package.

    Parameters
    ----------
    measure_in_process : bool, True by default
        Mesure memory allocation in the current process only, if false then measures at the system
        level.
    """

    if measure_in_process:
        process = psutil.Process()
        memory = process.memory_info().rss
    else:
        mem_info = psutil.virtual_memory()
        memory = mem_info.total - mem_info.available

    return memory


@pytest.mark.parametrize("mode", mode_list)
def test_lazy_random_recording(mode):
    bytes_to_MiB_factor = 1024**2
    relative_tolerance = 0.05  # relative tolerance of 5 per cent

    sampling_frequency = 30000  # Hz
    durations = [2.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0

    num_samples = int(durations[0] * sampling_frequency)
    # Around 100 MiB  4 bytes per sample * 384 channels * 30000  samples * 2 seconds duration
    expected_trace_size_MiB = dtype.itemsize * num_channels * num_samples / bytes_to_MiB_factor

    initial_memory_MiB = measure_memory_allocation() / bytes_to_MiB_factor
    lazy_recording = GeneratorRecording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        seed=seed,
        mode=mode,
    )

    memory_after_instanciation_MiB = measure_memory_allocation() / bytes_to_MiB_factor
    memory_after_instanciation_MiB == pytest.approx(initial_memory_MiB, rel=relative_tolerance)

    traces = lazy_recording.get_traces()
    expected_traces_shape = (int(durations[0] * sampling_frequency), num_channels)

    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert traces_size_MiB == expected_trace_size_MiB
    assert traces.shape == expected_traces_shape

    memory_after_traces_MiB = measure_memory_allocation() / bytes_to_MiB_factor

    print("Memory footprint")
    print(f"Initial memory = {initial_memory_MiB} MiB")
    print(f"Memory after instantiate class {memory_after_instanciation_MiB} MiB")
    print(f"Memory after traces {memory_after_traces_MiB} MiB")
    print(f"Traces size {traces_size_MiB} MiB")
    print(f"Difference between the last two {(memory_after_traces_MiB - traces_size_MiB)} MiB")

    (memory_after_instanciation_MiB + traces_size_MiB) == pytest.approx(memory_after_traces_MiB, rel=relative_tolerance)


@pytest.mark.parametrize("mode", mode_list)
def test_generate_lazy_recording(mode):
    bytes_to_MiB_factor = 1024**2
    full_traces_size_GiB = 1.0
    relative_tolerance = 0.05  # relative tolerance of 5 per cent

    initial_memory_MiB = measure_memory_allocation() / bytes_to_MiB_factor

    lazy_recording = generate_lazy_recording(full_traces_size_GiB=full_traces_size_GiB, mode=mode)

    memory_after_instanciation_MiB = measure_memory_allocation() / bytes_to_MiB_factor
    memory_after_instanciation_MiB == pytest.approx(initial_memory_MiB, rel=relative_tolerance)

    traces = lazy_recording.get_traces()
    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert full_traces_size_GiB * 1024 == traces_size_MiB

    memory_after_traces_MiB = measure_memory_allocation() / bytes_to_MiB_factor

    print("Memory footprint")
    print(f"Initial memory = {initial_memory_MiB} MiB")
    print(f"Memory after instantiate class {memory_after_instanciation_MiB} MiB")
    print(f"Memory after traces {memory_after_traces_MiB} MiB")
    print(f"Traces size {traces_size_MiB} MiB")
    print(f"Difference between the last two {(memory_after_traces_MiB - traces_size_MiB)} MiB")

    (memory_after_instanciation_MiB + traces_size_MiB) == pytest.approx(memory_after_traces_MiB, rel=relative_tolerance)


@pytest.mark.parametrize("mode", mode_list)
def test_generate_lazy_recording_under_giga(mode):
    recording = generate_lazy_recording(full_traces_size_GiB=0.5, mode=mode)
    assert recording.get_memory_size() == "512.00 MiB"


@pytest.mark.parametrize("mode", mode_list)
def test_generate_recording_correct_sizes(mode):
    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0

    lazy_recording = GeneratorRecording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        seed=seed,
        mode=mode,
    )

    num_frames = lazy_recording.get_num_frames(segment_index=0)
    assert num_frames == sampling_frequency * durations[0]

    traces = lazy_recording.get_traces()

    assert traces.shape == (num_frames, num_channels)


@pytest.mark.parametrize("mode", mode_list)
def test_generator_recording_consistency(mode):
    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0

    lazy_recording = GeneratorRecording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        seed=seed,
        mode=mode,
    )

    traces = lazy_recording.get_traces(start_frame=0, end_frame=None)
    assert np.allclose(lazy_recording.get_traces(), traces)

    start_frame = 25
    end_frame = 80
    traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    same_traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    assert np.allclose(traces, same_traces)


@pytest.mark.parametrize("mode", mode_list)
def test_generator_recording_consistency_across_traces(mode):
    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0

    lazy_recording = GeneratorRecording(
        durations=durations,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
        seed=seed,
        mode=mode,
    )

    extra_samples = 10
    start_frame = 0
    end_frame = 1000
    traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    lager_traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame + extra_samples)
    assert np.allclose(traces, lager_traces[:end_frame, :])
