import pytest
import psutil

import numpy as np
from spikeinterface.core.generate import LazyRandomRecording, generate_specific_size_recording


def test_lazy_random_recording():
    bytes_to_MiB_factor = 1024 * 1024
    
    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0
    
    num_samples = int(durations[0] * sampling_frequency)
    # Around 50 MiB                  4 bytes per sample * 384 channels * 30000 samples
    expected_trace_size_MiB = dtype.itemsize * num_channels * num_samples / bytes_to_MiB_factor
    print(expected_trace_size_MiB)  # This is around 1MB
    
    process = psutil.Process()
    initial_memory_MiB = process.memory_info().rss / bytes_to_MiB_factor

    lazy_recording = LazyRandomRecording(
        durations=durations, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype, seed=seed
    )
    memory_after_instanciation_MiB = process.memory_info().rss / bytes_to_MiB_factor
    assert round(memory_after_instanciation_MiB) == round(initial_memory_MiB)

    traces = lazy_recording.get_traces()
    expected_traces_shape = (int(durations[0] * sampling_frequency), num_channels)
    
    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert traces_size_MiB == expected_trace_size_MiB
    assert traces.shape == expected_traces_shape
    
    memory_after_traces_MiB = process.memory_info().rss / bytes_to_MiB_factor
    assert round(memory_after_traces_MiB) == round(memory_after_instanciation_MiB + traces_size_MiB)


def test_generate_large_recording():
    bytes_to_GiB_factor = 1024 ** 3
    process = psutil.Process()
    initial_memory_GiB = process.memory_info().rss / bytes_to_GiB_factor 
    full_traces_size_GiB = 1.0
    recording = generate_specific_size_recording(full_traces_size_GiB = full_traces_size_GiB)
    memory_after_instanciation_GiB = process.memory_info().rss / bytes_to_GiB_factor
    assert round(memory_after_instanciation_GiB) == round(initial_memory_GiB)
    
    traces = recording.get_traces()
    assert full_traces_size_GiB == round(traces.nbytes / bytes_to_GiB_factor) 
    
    memory_after_traces_GiB = process.memory_info().rss / bytes_to_GiB_factor
    assert round(memory_after_traces_GiB) == round(memory_after_instanciation_GiB + full_traces_size_GiB)
