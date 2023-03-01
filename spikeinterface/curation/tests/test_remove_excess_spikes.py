import numpy as np
from spikeinterface.core.generate import generate_recording, NumpySorting
from spikeinterface.core.waveform_tools import has_exceeding_spikes
from spikeinterface.curation import remove_excess_spikes


def test_remove_excess_spikes():
    durations = [5, 4]
    sampling_frequency = 30000
    recording = generate_recording(durations=durations, sampling_frequency=sampling_frequency)

    # create two units with excess 5 excess spikes each in each segment
    num_units = 2
    num_spikes = 100
    num_excess_spikes_per_segment = 5
    times = []
    labels = []
    for segment_index in range(recording.get_num_segments()):
        num_samples = recording.get_num_samples(segment_index=segment_index)
        times_segment = np.array([], dtype=int)
        labels_segment = np.array([], dtype=int)
        for unit in range(num_units):
            spike_times = np.random.randint(0, num_samples - 1, num_spikes)
            excess_spikes = np.random.randint(num_samples, num_samples + 100, num_excess_spikes_per_segment)
            spike_times = np.sort(np.concatenate((spike_times, excess_spikes)))
            spike_labels = unit * np.ones_like(spike_times)
            times_segment = np.concatenate((times_segment, spike_times))
            labels_segment = np.concatenate((labels_segment, spike_labels))
        times.append(times_segment)
        labels.append(labels_segment)

    sorting = NumpySorting.from_times_labels(times, labels, sampling_frequency=sampling_frequency)
    assert has_exceeding_spikes(recording, sorting)

    sorting_excess = remove_excess_spikes(sorting, recording)
    assert not has_exceeding_spikes(recording, sorting_excess)


if __name__ == "__main__":
    test_remove_excess_spikes()
