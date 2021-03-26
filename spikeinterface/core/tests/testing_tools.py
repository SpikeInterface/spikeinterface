import numpy as np

from spikeinterface import NumpyRecording

from probeinterface import generate_linear_probe

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
    
    probe = generate_linear_probe(num_elec=num_channels)
    probe.set_device_channel_indices(np.arange(num_channels))
    recording.set_probe(probe, in_place=True)

    return recording
    


def create_sorting_npz(num_seg, file_path):
    # create a NPZ sorting file
    d = {}
    d['unit_ids'] = np.array([0,1,2], dtype='int64')
    d['num_segment'] = np.array([2], dtype='int64')
    d['sampling_frequency'] = np.array([30000.], dtype='float64')
    for seg_index in range(num_seg):
        spike_indexes = np.arange(0, 1000, 10)
        spike_labels = np.zeros(spike_indexes.size, dtype='int64')
        spike_labels[0::3] = 0
        spike_labels[1::3] = 1
        spike_labels[2::3] = 2
        d[f'spike_indexes_seg{seg_index}'] = spike_indexes
        d[f'spike_labels_seg{seg_index}'] = spike_labels
    np.savez(file_path, **d)


if __name__ == '__main__':
    print(generate_recording())    
