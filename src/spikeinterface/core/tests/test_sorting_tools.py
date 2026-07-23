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
    remap_unit_indices_in_vector,
    reorder_spike_vector_by_unit_and_segment,
)
from spikeinterface.core.base import minimum_spike_dtype


@pytest.fixture(params=[True, False], ids=["numba", "numpy"])
def force_numba(request, monkeypatch):
    """Run each test once with numba enabled (if installed) and once with the numpy fallback."""
    if request.param and importlib.util.find_spec("numba") is None:
        pytest.skip("numba not installed")
    monkeypatch.setattr("spikeinterface.core.sorting_tools.HAVE_NUMBA", request.param)
    return request.param


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


def _make_spike_vector(sample_indices, unit_indices, segment_indices):
    spikes = np.empty(len(sample_indices), dtype=minimum_spike_dtype)
    spikes["sample_index"] = sample_indices
    spikes["unit_index"] = unit_indices
    spikes["segment_index"] = segment_indices
    return spikes


def test_reorder_spike_vector_by_unit_and_segment(force_numba):
    # 3 units, 1 segment, so the output is simply grouped by unit.
    spikes = _make_spike_vector(
        sample_indices=[10, 10, 11, 12, 12, 13],
        unit_indices=[2, 0, 1, 2, 0, 0],
        segment_indices=[0, 0, 0, 0, 0, 0],
    )

    ordered_spikes, order, counts = reorder_spike_vector_by_unit_and_segment(spikes, 3, 1)

    assert np.array_equal(counts, [3, 1, 2])
    assert np.array_equal(spikes[order], ordered_spikes)
    assert np.array_equal(ordered_spikes["unit_index"], [0, 0, 0, 1, 2, 2])
    # Stability: within each bucket, sample_index keeps its (ascending) input order.
    assert np.array_equal(ordered_spikes["sample_index"], [10, 12, 13, 11, 10, 12])


def test_reorder_spike_vector_by_unit_and_segment_raises(force_numba):
    """Out-of-range indices must raise on both paths, rather than write out of bounds."""
    spikes = _make_spike_vector([0, 1, 2], [0, 1, 0], 0)

    with pytest.raises(ValueError, match="must not be negative"):
        reorder_spike_vector_by_unit_and_segment(spikes, -1, 1)

    with pytest.raises(ValueError, match="outside"):
        reorder_spike_vector_by_unit_and_segment(spikes, 1, 1)  # unit_index 1 >= num_units
    with pytest.raises(ValueError, match="outside"):
        reorder_spike_vector_by_unit_and_segment(_make_spike_vector([0], [0], [5]), 1, 1)


@pytest.mark.parametrize("num_units", [2, 300, 70_000], ids=["uint8", "uint16", "uint32"])
def test_reorder_spike_vector_by_unit_and_segment_bucket_dtypes(monkeypatch, num_units):
    """The numpy path narrows the bucket dtype to num_buckets; every width must stay correct."""
    monkeypatch.setattr("spikeinterface.core.sorting_tools.HAVE_NUMBA", False)
    rng = np.random.default_rng(0)
    num_spikes = 1_000
    spikes = _make_spike_vector(
        sample_indices=np.arange(num_spikes),
        unit_indices=rng.integers(0, num_units, size=num_spikes),
        segment_indices=0,
    )
    ordered_spikes, order, counts = reorder_spike_vector_by_unit_and_segment(spikes, num_units, 1)
    assert np.array_equal(spikes[order], ordered_spikes)
    assert np.array_equal(ordered_spikes, spikes[np.argsort(spikes["unit_index"], kind="stable")])
    assert counts.sum() == num_spikes


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
    sorting1 = NumpySorting.from_samples_and_labels(
        [times, times], [labels, labels], 10_000.0, unit_ids=["a", "b", "c"]
    )
    spikes1 = sorting1.to_spike_vector()

    sorting2 = apply_merges_to_sorting(sorting1, [["a", "b"]], censor_ms=None)
    spikes2 = sorting2.to_spike_vector()
    assert sorting2.unit_ids.size == 2
    assert sorting1.to_spike_vector().size == sorting2.to_spike_vector().size
    assert np.array_equal(["c", "merge0"], sorting2.unit_ids)
    assert np.array_equal(
        spikes1[spikes1["unit_index"] == 2]["sample_index"], spikes2[spikes2["unit_index"] == 0]["sample_index"]
    )

    sorting3, keep_mask, _ = apply_merges_to_sorting(sorting1, [["a", "b"]], censor_ms=1.5, return_extra=True)
    spikes3 = sorting3.to_spike_vector()
    assert spikes3.size < spikes1.size
    assert not keep_mask[1]
    st = sorting3.get_unit_spike_train(segment_index=0, unit_id="merge0")
    assert st.size == 3  # one spike is removed by censor period

    # unit_ids int
    sorting1 = NumpySorting.from_samples_and_labels([times, times], [labels, labels], 10_000.0, unit_ids=[10, 20, 30])
    spikes1 = sorting1.to_spike_vector()
    sorting2 = apply_merges_to_sorting(sorting1, [[10, 20]], censor_ms=None)
    assert np.array_equal(sorting2.unit_ids, [30, 31])

    sorting1 = NumpySorting.from_samples_and_labels(
        [times, times], [labels, labels], 10_000.0, unit_ids=["a", "b", "c"]
    )
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

    new_unit_ids = generate_unit_ids_for_merge_group(
        ["0", "5", "12", "9", "15"], [["0", "5"], ["9", "15"]], new_id_strategy="join"
    )
    assert np.array_equal(new_unit_ids, ["0-5", "9-15"])


def test_remap_unit_indices_in_vector():

    unit_ids = ["a", "b", "c", "d", "e"]
    n_spikes = 20
    n_units = len(unit_ids)

    spikes = np.zeros(n_spikes, dtype=minimum_spike_dtype)
    spikes["unit_index"] = np.arange(n_spikes) % n_units
    # the sample should remain the original unit_index after transform
    spikes["sample_index"] = np.arange(n_spikes) % n_units
    # print(spikes)

    # remove some units
    # so 0->0, 2->1, 4->2
    new_unit_ids = ["a", "c", "e"]
    new_spikes, mask = remap_unit_indices_in_vector(spikes, unit_ids, new_unit_ids, keep_old_unit_ids=None)
    assert np.all(np.isin(new_spikes["unit_index"], [0, 1, 2]))
    assert new_spikes.size == n_spikes * len(new_unit_ids) // n_units
    # print(new_spikes)

    # rename units in reverse order
    # so 0->4, 1->3, 2->2, 3->1,  4->0
    new_unit_ids = ["e", "d", "c", "b", "a"]
    new_spikes, mask = remap_unit_indices_in_vector(spikes, unit_ids, new_unit_ids, keep_old_unit_ids=None)
    assert new_spikes.size == spikes.size
    assert np.all(new_spikes["unit_index"] == 4 - new_spikes["sample_index"])
    # print(new_spikes)

    # add some new units
    # vector unchanged
    new_unit_ids = ["a", "b", "c", "d", "e", "f", "g"]
    new_spikes, mask = remap_unit_indices_in_vector(spikes, unit_ids, new_unit_ids, keep_old_unit_ids=None)
    assert np.array_equal(new_spikes, spikes)
    # print(new_spikes)

    # add some + remove some
    # so 0->0, 2->1, 4->2
    new_unit_ids = ["a", "c", "e", "f", "g"]
    new_spikes, mask = remap_unit_indices_in_vector(spikes, unit_ids, new_unit_ids, keep_old_unit_ids=None)
    assert np.all(np.isin(new_spikes["unit_index"], [0, 1, 2]))
    assert new_spikes.size == n_spikes * 3 // n_units
    # print(new_spikes)

    # remove one unit which is also in the new unit set
    # the unit_id="e" (index=4) will not be in new set
    new_unit_ids = ["a", "b", "c", "d", "e"]
    keep_old_unit_ids = ["a", "b", "c", "d"]
    new_spikes, mask = remap_unit_indices_in_vector(spikes, unit_ids, new_unit_ids, keep_old_unit_ids=keep_old_unit_ids)
    assert np.all(np.isin(new_spikes["unit_index"], [0, 1, 2, 3]))
    assert new_spikes.size == n_spikes * 4 // n_units
    target_mask = np.ones(spikes.size, dtype=bool)
    target_mask[4::5] = False
    assert np.array_equal(mask, target_mask)


if __name__ == "__main__":
    # test_spike_vector_to_spike_trains()
    # test_spike_vector_to_indices()
    # test_random_spikes_selection()

    # test_apply_merges_to_sorting()
    # test_get_ids_after_merging()
    # test_generate_unit_ids_for_merge_group()

    test_remap_unit_indices_in_vector()
