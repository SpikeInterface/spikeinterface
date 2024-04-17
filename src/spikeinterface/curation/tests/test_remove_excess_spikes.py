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
    num_num_samples_spikes_per_segment = 5
    num_excess_spikes_per_segment = 5
    num_neg_spike_times_per_segment = 2
    times = []
    labels = []
    for segment_index in range(recording.get_num_segments()):
        num_samples = recording.get_num_samples(segment_index=segment_index)
        times_segment = np.array([], dtype=int)
        labels_segment = np.array([], dtype=int)
        for unit in range(num_units):
            neg_spike_times = np.random.randint(-50, -1, num_neg_spike_times_per_segment)
            spike_times = np.random.randint(0, num_samples, num_spikes)
            last_samples_spikes = (num_samples - 1) * np.ones(num_num_samples_spikes_per_segment, dtype=int)
            num_samples_spike_times = num_samples * np.ones(num_num_samples_spikes_per_segment, dtype=int)
            excess_spikes = np.random.randint(num_samples, num_samples + 100, num_excess_spikes_per_segment)
            spike_times = np.sort(
                np.concatenate(
                    (neg_spike_times, spike_times, last_samples_spikes, num_samples_spike_times, excess_spikes)
                )
            )
            spike_labels = unit * np.ones_like(spike_times)
            times_segment = np.concatenate((times_segment, spike_times))
            labels_segment = np.concatenate((labels_segment, spike_labels))
        times.append(times_segment)
        labels.append(labels_segment)

    sorting = NumpySorting.from_times_labels(times, labels, sampling_frequency=sampling_frequency)
    assert has_exceeding_spikes(recording, sorting)

    sorting_corrected = remove_excess_spikes(sorting, recording)
    assert not has_exceeding_spikes(recording, sorting_corrected)

    for u in sorting.unit_ids:
        for segment_index in range(sorting.get_num_segments()):
            spike_train_excess = sorting.get_unit_spike_train(u, segment_index=segment_index)
            spike_train_corrected = sorting_corrected.get_unit_spike_train(u, segment_index=segment_index)

            assert (
                len(spike_train_corrected)
                == len(spike_train_excess)
                - num_num_samples_spikes_per_segment
                - num_excess_spikes_per_segment
                - num_neg_spike_times_per_segment
            )


if __name__ == "__main__":
    test_remove_excess_spikes()
