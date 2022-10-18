import numpy as np
from spikeinterface.core.testing_tools import generate_sorting
from spikeinterface.curation import remove_duplicated_spikes, find_duplicated_spikes_random, find_duplicated_spikes_keep_first, find_duplicated_spikes_keep_last


def test_remove_duplicated_spikes() -> None:
	sorting = generate_sorting()

	censored_period_ms = 0.5
	censored_period = int(round(0.5 * 1e-3 * sorting.get_sampling_frequency()))

	for method in ("keep_first", "keep_last", "random"):
		new_sorting = remove_duplicated_spikes(sorting, censored_period_ms, method=method)

		for segment_index in range(sorting.get_num_segments()):
			for unit_id in sorting.unit_ids:
				assert len(sorting.get_unit_spike_train(unit_id, segment_index=segment_index)) >= \
					   len(new_sorting.get_unit_spike_train(unit_id, segment_index=segment_index))

				assert np.all(np.diff(new_sorting.get_unit_spike_train(unit_id, segment_index=segment_index)) > censored_period)


def test_find_duplicated_spikes() -> None:
	spike_train = np.array([20, 80, 81, 150, 152, 900], dtype=np.int64)

	assert len(find_duplicated_spikes_random(spike_train, censored_period=5)) == 2
	assert len(find_duplicated_spikes_keep_first(spike_train, censored_period=5)) == 2
	assert len(find_duplicated_spikes_keep_last(spike_train, censored_period=5)) == 2
