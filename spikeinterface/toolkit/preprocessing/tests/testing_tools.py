
import numpy as np

from spikeinterface import NumpyRecording

def generate_recording(
        num_channels = 2,
        sampling_frequency = 30000.,  # in Hz
        durations = [10.325, 3.5], # in s for 2 segments
    ):
    
    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]
    
    traces_list = []
    for i in range(num_segments):
        traces = np.random.randn(num_timepoints[i], num_channels).astype('float32')
        times = np.arange(num_timepoints[i])  / sampling_frequency
        traces += np.sin(2*np.pi*50*times)[:, None]
        traces_list.append(traces)
    recording = NumpyRecording(traces_list, sampling_frequency)
    
    return recording
    
if __name__ == '__main__':
    print(generate_recording())

