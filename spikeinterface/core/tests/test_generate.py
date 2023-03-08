import pytest
import psutil



import numpy as np
from spikeinterface.core.generate import LazyRandomRecording, generate_lazy_random_recording


def test_lazy_random_recording_process():
    bytes_to_MiB_factor = 1024 ** 2
    
    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0
    
    num_samples = int(durations[0] * sampling_frequency)
    # Around 50 MiB  4 bytes per sample * 384 channels * 30000 samples
    expected_trace_size_MiB = dtype.itemsize * num_channels * num_samples / bytes_to_MiB_factor
    
    process = psutil.Process()
    initial_memory_MiB = process.memory_info().rss / bytes_to_MiB_factor

    lazy_recording = LazyRandomRecording(
        durations=durations, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype, seed=seed
    )
    memory_after_instanciation_MiB = process.memory_info().rss / bytes_to_MiB_factor
    excess_memory = memory_after_instanciation_MiB / initial_memory_MiB
    assert excess_memory == pytest.approx(1.0, rel=1e-2)  

    traces = lazy_recording.get_traces()
    expected_traces_shape = (int(durations[0] * sampling_frequency), num_channels)
    
    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert traces_size_MiB == expected_trace_size_MiB
    assert traces.shape == expected_traces_shape
    
    memory_after_traces_MiB = process.memory_info().rss / bytes_to_MiB_factor
    excess_memory = memory_after_traces_MiB / (memory_after_instanciation_MiB + traces_size_MiB)

    print("Memory usage")
    print(f"{initial_memory_MiB=} - {memory_after_instanciation_MiB=} - {traces_size_MiB=} - {memory_after_traces_MiB=}")

    assert excess_memory == pytest.approx(1.0, rel=1e-2)


def test_lazy_random_recording():
    bytes_to_MiB_factor = 1024 ** 2
    rel = 0.1  # relative tolerance

    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0
    
    num_samples = int(durations[0] * sampling_frequency)
    # Around 50 MiB  4 bytes per sample * 384 channels * 30000 samples
    expected_trace_size_MiB = dtype.itemsize * num_channels * num_samples / bytes_to_MiB_factor
    
    mem_info = psutil.virtual_memory()
    initial_memory_MiB = (mem_info.total - mem_info.available) / bytes_to_MiB_factor

    lazy_recording = LazyRandomRecording(
        durations=durations, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype, seed=seed
    )
    
    mem_info = psutil.virtual_memory()
    memory_after_instanciation_MiB = (mem_info.total - mem_info.available) / bytes_to_MiB_factor
    excess_memory = memory_after_instanciation_MiB / initial_memory_MiB
    assert excess_memory == pytest.approx(1.0, rel=rel)  

    traces = lazy_recording.get_traces()
    expected_traces_shape = (int(durations[0] * sampling_frequency), num_channels)
    
    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert traces_size_MiB == expected_trace_size_MiB
    assert traces.shape == expected_traces_shape
    
    mem_info = psutil.virtual_memory()
    memory_after_traces_MiB = (mem_info.total - mem_info.available) / bytes_to_MiB_factor
    excess_memory = memory_after_traces_MiB / (memory_after_instanciation_MiB + traces_size_MiB)

    assert excess_memory == pytest.approx(1.0, rel=rel)
    
    print("Memory usage")
    print(f"{initial_memory_MiB=} - {memory_after_instanciation_MiB=} - {traces_size_MiB=} - {memory_after_traces_MiB=}")


def test_generate_large_recording():
    bytes_to_GiB_factor = 1024 ** 3
    full_traces_size_GiB = 1.0
    rel = 0.1  # Relative tolerance on error

    mem_info = psutil.virtual_memory()
    initial_memory_GiB = (mem_info.total - mem_info.available) / bytes_to_GiB_factor 
    recording = generate_lazy_random_recording(full_traces_size_GiB = full_traces_size_GiB)
    
    mem_info = psutil.virtual_memory()
    memory_after_instanciation_GiB = (mem_info.total - mem_info.available) / bytes_to_GiB_factor
    
    excess_memory = memory_after_instanciation_GiB / initial_memory_GiB
    assert excess_memory == pytest.approx(1.0, rel=rel) 

    traces = recording.get_traces()
    assert full_traces_size_GiB == round(traces.nbytes / bytes_to_GiB_factor) 
    
    mem_info = psutil.virtual_memory()
    memory_after_traces_GiB = (mem_info.total - mem_info.available) / bytes_to_GiB_factor
    excess_memory = memory_after_traces_GiB  / (memory_after_instanciation_GiB + full_traces_size_GiB)
    assert excess_memory == pytest.approx(1.0, rel=rel)  

    print("Memory usage")
    print(f"{initial_memory_GiB=} - {memory_after_instanciation_GiB=} - {full_traces_size_GiB=} - {memory_after_traces_GiB=}")