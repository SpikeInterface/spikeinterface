import numpy as np

from spikeinterface import NumpyRecording, NumpySorting
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from probeinterface import generate_linear_probe
from spikeinterface.core.snippets_tools import snippets_from_sorting

def generate_recording(
        num_channels=2,
        sampling_frequency=30000.,  # in Hz
        durations=[10.325, 3.5],  #  in s for 2 segments
        set_probe=True,
        ndim=2
):
    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]

    traces_list = []
    for i in range(num_segments):
        traces = np.random.randn(num_timepoints[i], num_channels).astype('float32')
        times = np.arange(num_timepoints[i]) / sampling_frequency
        traces += np.sin(2 * np.pi * 50 * times)[:, None]
        traces_list.append(traces)
    recording = NumpyRecording(traces_list, sampling_frequency)

    if set_probe:
        probe = generate_linear_probe(num_elec=num_channels)
        if ndim == 3:
            probe = probe.to_3d()
        probe.set_device_channel_indices(np.arange(num_channels))
        recording.set_probe(probe, in_place=True)
        probe = generate_linear_probe(num_elec=num_channels)
    return recording


def generate_sorting(
        num_units=5,
        sampling_frequency=30000.,  # in Hz
        durations=[10.325, 3.5],  #  in s for 2 segments
        empty_units=None
):
    num_segments = len(durations)
    num_timepoints = [int(sampling_frequency * d) for d in durations]

    unit_ids = np.arange(num_units)

    if empty_units is None:
        empty_units = []

    units_dict_list = []
    for seg_index in range(num_segments):
        units_dict = {}
        for unit_id in unit_ids:
            if unit_id not in empty_units:
                #  15 Hz for all units
                n_spike = int(15. * durations[seg_index])
                spike_times = np.sort(np.unique(np.random.randint(0, num_timepoints[seg_index], n_spike)))
                units_dict[unit_id] = spike_times
            else:
                units_dict[unit_id] = np.array([], dtype=int)
        units_dict_list.append(units_dict)
    sorting = NumpySorting.from_dict(units_dict_list, sampling_frequency)

    return sorting


def create_sorting_npz(num_seg, file_path):
    # create a NPZ sorting file
    d = {}
    d['unit_ids'] = np.array([0, 1, 2], dtype='int64')
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


def generate_snippets(
        nbefore=20,
        nafter=44,
        num_channels=2,
        wf_folder=None,
        sampling_frequency=30000.,  # in Hz
        durations=[10.325, 3.5],  #  in s for 2 segments
        set_probe=True,
        ndim=2,
        num_units=5,
        empty_units=None, **job_kwargs
):

    recording = generate_recording(durations=durations, num_channels=num_channels,
                                   sampling_frequency=sampling_frequency, ndim=ndim,
                                   set_probe=set_probe)

    sorting = generate_sorting(num_units=num_units,  sampling_frequency=sampling_frequency,
                               durations=durations, empty_units=empty_units)

    snippets = snippets_from_sorting(recording=recording, sorting=sorting,
        nbefore=nbefore, nafter=nafter, wf_folder=wf_folder, **job_kwargs)
    
    if set_probe:
        probe = recording.get_probe()
        snippets = snippets.set_probe(probe)

    return snippets, sorting


if __name__ == '__main__':
    print(generate_recording())
    print(generate_sorting())
    print(generate_snippets())
