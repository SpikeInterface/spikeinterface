import importlib
import pytest
import numpy as np

from spikeinterface.core import NumpySorting

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core.sorting_tools import (
    spike_vector_to_spike_trains,
    random_spikes_selection,
    spike_vector_to_indices,
)


@pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="Testing `spike_vector_to_dict` requires Python package 'numba'."
)
def test_spike_vector_to_spike_trains():
    sorting = NumpySorting.from_unit_dict({1: np.array([0, 51, 108]), 5: np.array([23, 87])}, 30_000)
    spike_vector = sorting.to_spike_vector(concatenated=False)
    spike_trains = spike_vector_to_spike_trains(spike_vector, sorting.unit_ids)

    assert len(spike_trains[0]) == sorting.get_num_units()
    for unit_index, unit_id in enumerate(sorting.unit_ids):
        assert np.array_equal(spike_trains[0][unit_id], sorting.get_unit_spike_train(unit_id=unit_id, segment_index=0))


def test_spike_vector_to_indices():
    sorting = NumpySorting.from_unit_dict({1: np.array([0, 51, 108]), 5: np.array([23, 87])}, 30_000)
    spike_vector = sorting.to_spike_vector(concatenated=False)
    spike_indices = spike_vector_to_indices(spike_vector, sorting.unit_ids)

    segment_index = 0
    assert len(spike_indices[segment_index]) == sorting.get_num_units()
    for unit_index, unit_id in enumerate(sorting.unit_ids):
        inds = spike_indices[segment_index][unit_id]
        assert np.array_equal(
            spike_vector[segment_index][inds]["sample_index"],
            sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index),
        )


def test_random_spikes_selection():
    recording, sorting = generate_ground_truth_recording(
        durations=[20.0, 10.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    max_spikes_per_unit = 12
    num_samples = [recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())]

    random_spikes_indices = random_spikes_selection(
        sorting, num_samples, method="uniform", max_spikes_per_unit=max_spikes_per_unit, margin_size=None, seed=2205
    )
    random_spikes_indices1 = random_spikes_indices
    spikes = sorting.to_spike_vector()
    some_spikes = spikes[random_spikes_indices]
    for unit_index, unit_id in enumerate(sorting.unit_ids):
        spike_slected_unit = some_spikes[some_spikes["unit_index"] == unit_index]
        assert spike_slected_unit.size == max_spikes_per_unit

    # with margin
    random_spikes_indices = random_spikes_selection(
        sorting, num_samples, method="uniform", max_spikes_per_unit=max_spikes_per_unit, margin_size=25, seed=2205
    )
    # in that case the number is not garanty so it can be a bit less
    assert random_spikes_indices.size >= (0.9 * sorting.unit_ids.size * max_spikes_per_unit)

    # all
    random_spikes_indices = random_spikes_selection(sorting, num_samples, method="all")
    assert random_spikes_indices.size == spikes.size


if __name__ == "__main__":
    # test_spike_vector_to_spike_trains()
    # test_spike_vector_to_indices()
    test_random_spikes_selection()
