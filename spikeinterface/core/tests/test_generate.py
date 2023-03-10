import pytest
import psutil

import numpy as np

from spikeinterface.core.generate import GeneratorRecording, generate_lazy_recording

def measure_memory_allocation(measure_in_process: bool=True) -> float:
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

def test_lazy_random_recording():
    bytes_to_MiB_factor = 1024 ** 2
    rel = 0.05  # relative tolerance

    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0
    
    num_samples = int(durations[0] * sampling_frequency)
    # Around 50 MiB  4 bytes per sample * 384 channels * 30000 samples
    expected_trace_size_MiB = dtype.itemsize * num_channels * num_samples / bytes_to_MiB_factor
    

    initial_memory_MiB = measure_memory_allocation() / bytes_to_MiB_factor
    lazy_recording = GeneratorRecording(
        durations=durations, sampling_frequency=sampling_frequency, num_channels=num_channels, dtype=dtype, seed=seed
    )
    
    memory_after_instanciation_MiB = measure_memory_allocation() / bytes_to_MiB_factor

    excess_memory = memory_after_instanciation_MiB / initial_memory_MiB
    assert excess_memory == pytest.approx(1.0, rel=rel)  

    traces = lazy_recording.get_traces()
    expected_traces_shape = (int(durations[0] * sampling_frequency), num_channels)
    
    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert traces_size_MiB == expected_trace_size_MiB
    assert traces.shape == expected_traces_shape

    memory_after_traces_MiB = measure_memory_allocation() / bytes_to_MiB_factor    
    excess_memory = memory_after_traces_MiB / (memory_after_instanciation_MiB + traces_size_MiB)

    print("Memory footprint")
    print(f"Initial memory = {initial_memory_MiB}")
    print(f"Memory after instantiate class {memory_after_instanciation_MiB}")
    print(f"Memory after traces {memory_after_traces_MiB}")
    print(f"Traces size {traces_size_MiB}")
    print(f"Difference between the last two", {(memory_after_traces_MiB - traces_size_MiB)})
    
    assert excess_memory == pytest.approx(1.0, rel=rel)
    

def test_generate_lazy_recording():
    
    bytes_to_MiB_factor = 1024 ** 2
    full_traces_size_GiB = 1.0
    rel = 0.05  # relative tolerance

    initial_memory_MiB = measure_memory_allocation() / bytes_to_MiB_factor

    lazy_recording = generate_lazy_recording(full_traces_size_GiB = full_traces_size_GiB)
    memory_after_instanciation_MiB = measure_memory_allocation() / bytes_to_MiB_factor
        
    excess_memory = memory_after_instanciation_MiB / initial_memory_MiB
    assert excess_memory == pytest.approx(1.0, rel=rel) 

    traces = lazy_recording.get_traces()
    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert full_traces_size_GiB * 1024 == traces_size_MiB

    memory_after_traces_MiB = measure_memory_allocation() / bytes_to_MiB_factor
        
    excess_memory = memory_after_traces_MiB  / (memory_after_instanciation_MiB + traces_size_MiB)


    print("Memory footprint")
    print(f"Initial memory = {initial_memory_MiB}")
    print(f"Memory after instantiate class {memory_after_instanciation_MiB}")
    print(f"Memory after traces {memory_after_traces_MiB}")
    print(f"Traces size {traces_size_MiB}")
    print(f"Difference between the last two", {(memory_after_traces_MiB - traces_size_MiB)})