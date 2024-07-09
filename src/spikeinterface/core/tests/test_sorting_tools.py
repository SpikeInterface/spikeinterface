import importlib
import pytest
import numpy as np

from spikeinterface.core import NumpySorting

from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core.sorting_tools import (
    spike_vector_to_spike_trains,
    random_spikes_selection,
    spike_vector_to_indices,
    apply_merges_to_sorting,
    _get_ids_after_merging,
    generate_unit_ids_for_merge_group,
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


def test_apply_merges_to_sorting():

    times = np.array([0, 0, 10, 20, 300])
    labels = np.array(["a", "b", "c", "a", "b"])

    # unit_ids str
    sorting1 = NumpySorting.from_times_labels([times, times], [labels, labels], 10_000.0, unit_ids=["a", "b", "c"])
    spikes1 = sorting1.to_spike_vector()

    sorting2 = apply_merges_to_sorting(sorting1, [["a", "b"]], censor_ms=None)
    spikes2 = sorting2.to_spike_vector()
    assert sorting2.unit_ids.size == 2
    assert sorting1.to_spike_vector().size == sorting1.to_spike_vector().size
    assert np.array_equal(["c", "merge0"], sorting2.unit_ids)
    assert np.array_equal(
        spikes1[spikes1["unit_index"] == 2]["sample_index"], spikes2[spikes2["unit_index"] == 0]["sample_index"]
    )

    sorting3, keep_mask = apply_merges_to_sorting(sorting1, [["a", "b"]], censor_ms=1.5, return_kept=True)
    spikes3 = sorting3.to_spike_vector()
    assert spikes3.size < spikes1.size
    assert not keep_mask[1]
    st = sorting3.get_unit_spike_train(segment_index=0, unit_id="merge0")
    assert st.size == 3  # one spike is removed by censor period

    # unit_ids int
    sorting1 = NumpySorting.from_times_labels([times, times], [labels, labels], 10_000.0, unit_ids=[10, 20, 30])
    spikes1 = sorting1.to_spike_vector()
    sorting2 = apply_merges_to_sorting(sorting1, [[10, 20]], censor_ms=None)
    assert np.array_equal(sorting2.unit_ids, [30, 31])

    sorting1 = NumpySorting.from_times_labels([times, times], [labels, labels], 10_000.0, unit_ids=["a", "b", "c"])
    sorting2 = apply_merges_to_sorting(sorting1, [["a", "b"]], censor_ms=None, new_id_strategy="take_first")
    assert np.array_equal(sorting2.unit_ids, ["a", "c"])


def test_get_ids_after_merging():

    all_unit_ids = _get_ids_after_merging(["a", "b", "c", "d", "e"], [["a", "b"], ["d", "e"]], ["x", "d"])
    assert np.array_equal(all_unit_ids, ["c", "d", "x"])
    # print(all_unit_ids)

    all_unit_ids = _get_ids_after_merging([0, 5, 12, 9, 15], [[0, 5], [9, 15]], [28, 9])
    assert np.array_equal(all_unit_ids, [12, 9, 28])
    # print(all_unit_ids)


def test_generate_unit_ids_for_merge_group():

    new_unit_ids = generate_unit_ids_for_merge_group(
        ["a", "b", "c", "d", "e"], [["a", "b"], ["d", "e"]], new_id_strategy="append"
    )
    assert np.array_equal(new_unit_ids, ["merge0", "merge1"])

    new_unit_ids = generate_unit_ids_for_merge_group(
        ["a", "b", "c", "d", "e"], [["a", "b"], ["d", "e"]], new_id_strategy="take_first"
    )
    assert np.array_equal(new_unit_ids, ["a", "d"])

    new_unit_ids = generate_unit_ids_for_merge_group([0, 5, 12, 9, 15], [[0, 5], [9, 15]], new_id_strategy="append")
    assert np.array_equal(new_unit_ids, [16, 17])

    new_unit_ids = generate_unit_ids_for_merge_group([0, 5, 12, 9, 15], [[0, 5], [9, 15]], new_id_strategy="take_first")
    assert np.array_equal(new_unit_ids, [0, 9])

    new_unit_ids = generate_unit_ids_for_merge_group(
        ["0", "5", "12", "9", "15"], [["0", "5"], ["9", "15"]], new_id_strategy="append"
    )
    assert np.array_equal(new_unit_ids, ["16", "17"])

    new_unit_ids = generate_unit_ids_for_merge_group(
        ["0", "5", "12", "9", "15"], [["0", "5"], ["9", "15"]], new_id_strategy="take_first"
    )
    assert np.array_equal(new_unit_ids, ["0", "9"])


if __name__ == "__main__":
    # test_spike_vector_to_spike_trains()
    # test_spike_vector_to_indices()
    # test_random_spikes_selection()

    test_apply_merges_to_sorting()
    test_get_ids_after_merging()
    test_generate_unit_ids_for_merge_group()
