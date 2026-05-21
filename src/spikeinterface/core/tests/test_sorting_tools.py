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
    is_spike_vector_sorted,
    build_spike_vector_from_sorted_arrays,
    filter_and_remap_spike_vector,
)
from spikeinterface.core.base import minimum_spike_dtype


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


def test_is_spike_vector_sorted():
    empty_spikes = np.zeros(0, dtype=minimum_spike_dtype)
    assert is_spike_vector_sorted(empty_spikes)

    one_spike = np.zeros(1, dtype=minimum_spike_dtype)
    assert is_spike_vector_sorted(one_spike)

    spikes = np.zeros(5, dtype=minimum_spike_dtype)
    spikes["segment_index"] = [0, 0, 1, 1, 1]
    spikes["sample_index"] = [100, 200, 0, 100, 100]
    spikes["unit_index"] = [0, 1, 0, 0, 1]
    assert is_spike_vector_sorted(spikes)
    assert is_spike_vector_sorted(spikes, chunk_size=None)
    assert is_spike_vector_sorted(spikes, chunk_size=1)

    segment_unsorted = spikes.copy()
    segment_unsorted["segment_index"] = [0, 1, 0, 1, 1]
    segment_unsorted["sample_index"] = [0, 100, 200, 300, 400]
    segment_unsorted["unit_index"] = [0, 0, 0, 0, 0]
    assert not is_spike_vector_sorted(segment_unsorted)

    sample_unsorted = spikes.copy()
    sample_unsorted["segment_index"] = 0
    sample_unsorted["sample_index"] = [0, 100, 50, 200, 300]
    sample_unsorted["unit_index"] = [0, 0, 0, 0, 0]
    assert not is_spike_vector_sorted(sample_unsorted)

    tie_unsorted = spikes.copy()
    tie_unsorted["segment_index"] = 0
    tie_unsorted["sample_index"] = [0, 100, 100, 200, 300]
    tie_unsorted["unit_index"] = [0, 1, 0, 0, 0]
    assert not is_spike_vector_sorted(tie_unsorted)

    with pytest.raises(ValueError, match="chunk_size"):
        is_spike_vector_sorted(spikes, chunk_size=0)


def test_is_spike_vector_sorted_chunk_boundaries():
    spikes = np.zeros(6, dtype=minimum_spike_dtype)
    spikes["segment_index"] = [0, 0, 1, 0, 1, 1]
    spikes["sample_index"] = [0, 100, 200, 300, 400, 500]
    spikes["unit_index"] = 0
    assert not is_spike_vector_sorted(spikes, chunk_size=3)

    spikes["segment_index"] = 0
    spikes["sample_index"] = [0, 100, 300, 200, 400, 500]
    assert not is_spike_vector_sorted(spikes, chunk_size=3)

    spikes["sample_index"] = [0, 100, 200, 200, 400, 500]
    spikes["unit_index"] = [0, 0, 1, 0, 0, 0]
    assert not is_spike_vector_sorted(spikes, chunk_size=3)


def test_is_spike_vector_sorted_assume_single_segment():
    spikes = np.zeros(5, dtype=minimum_spike_dtype)
    spikes["segment_index"] = [0, 1, 0, 1, 1]
    spikes["sample_index"] = [0, 100, 200, 300, 400]
    spikes["unit_index"] = [0, 0, 0, 0, 0]
    assert not is_spike_vector_sorted(spikes)
    assert is_spike_vector_sorted(spikes, assume_single_segment=True)

    sample_unsorted = spikes.copy()
    sample_unsorted["sample_index"] = [0, 100, 50, 200, 300]
    assert not is_spike_vector_sorted(sample_unsorted, assume_single_segment=True)

    tie_unsorted = spikes.copy()
    tie_unsorted["sample_index"] = [0, 100, 100, 200, 300]
    tie_unsorted["unit_index"] = [0, 1, 0, 0, 0]
    assert not is_spike_vector_sorted(tie_unsorted, assume_single_segment=True)


def _reference_spike_vector(sample_indices, unit_indices, segment_index=0):
    """Reference implementation: global lexsort, used as ground truth in tests."""
    n = sample_indices.size
    spikes = np.empty(n, dtype=minimum_spike_dtype)
    spikes["sample_index"] = sample_indices
    spikes["unit_index"] = unit_indices
    spikes["segment_index"] = segment_index
    order = np.lexsort((spikes["unit_index"], spikes["sample_index"]))
    return spikes[order]


@pytest.fixture(params=[True, False], ids=["numba", "numpy"])
def force_numba(request, monkeypatch):
    """Run each test once with numba enabled (if installed) and once with the fallback."""
    if request.param and importlib.util.find_spec("numba") is None:
        pytest.skip("numba not installed")
    monkeypatch.setattr("spikeinterface.core.sorting_tools.HAVE_NUMBA", request.param)
    return request.param


def test_build_spike_vector_no_ties(force_numba):
    sample_indices = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    unit_indices = np.array([3, 1, 4, 1, 5], dtype=np.int64)
    out = build_spike_vector_from_sorted_arrays(sample_indices, unit_indices)
    assert out.dtype == np.dtype(minimum_spike_dtype)
    assert np.array_equal(out["sample_index"], sample_indices)
    assert np.array_equal(out["unit_index"], unit_indices)
    assert np.all(out["segment_index"] == 0)


def test_build_spike_vector_with_ties(force_numba):
    # Three runs of ties (lengths 3, 2, 1, 4) with shuffled unit_indices
    sample_indices = np.array(
        [10, 10, 10, 20, 20, 30, 40, 40, 40, 40, 50],
        dtype=np.int64,
    )
    unit_indices = np.array([7, 2, 5, 9, 1, 3, 4, 0, 8, 2, 6], dtype=np.int64)
    out = build_spike_vector_from_sorted_arrays(sample_indices, unit_indices)
    ref = _reference_spike_vector(sample_indices, unit_indices)
    assert np.array_equal(out, ref)


def test_build_spike_vector_all_same_sample_index(force_numba):
    n = 64
    sample_indices = np.full(n, 42, dtype=np.int64)
    rng = np.random.default_rng(0)
    unit_indices = rng.permutation(n).astype(np.int64)
    out = build_spike_vector_from_sorted_arrays(sample_indices, unit_indices)
    ref = _reference_spike_vector(sample_indices, unit_indices)
    assert np.array_equal(out, ref)


def test_build_spike_vector_ties_at_edges(force_numba):
    # Ties at the very start, the very end, and an isolated single in between.
    sample_indices = np.array([5, 5, 5, 9, 12, 12, 12], dtype=np.int64)
    unit_indices = np.array([2, 0, 1, 7, 3, 1, 2], dtype=np.int64)
    out = build_spike_vector_from_sorted_arrays(sample_indices, unit_indices)
    ref = _reference_spike_vector(sample_indices, unit_indices)
    assert np.array_equal(out, ref)


def test_build_spike_vector_empty(force_numba):
    out = build_spike_vector_from_sorted_arrays(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )
    assert out.size == 0
    assert out.dtype == np.dtype(minimum_spike_dtype)


def test_build_spike_vector_segment_index(force_numba):
    sample_indices = np.array([0, 1, 2], dtype=np.int64)
    unit_indices = np.array([0, 0, 0], dtype=np.int64)
    out = build_spike_vector_from_sorted_arrays(sample_indices, unit_indices, segment_index=3)
    assert np.all(out["segment_index"] == 3)


def test_build_spike_vector_length_mismatch():
    with pytest.raises(ValueError):
        build_spike_vector_from_sorted_arrays(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
        )


def test_build_spike_vector_randomized_against_lexsort(force_numba):
    rng = np.random.default_rng(1234)
    n = 10_000
    # Build ~30% ties by drawing sample positions from a small space.
    sample_indices = np.sort(rng.integers(0, n // 3, size=n).astype(np.int64))
    unit_indices = rng.integers(0, 200, size=n).astype(np.int64)
    out = build_spike_vector_from_sorted_arrays(sample_indices, unit_indices)
    ref = _reference_spike_vector(sample_indices, unit_indices)
    assert np.array_equal(out, ref)


def test_build_spike_vector_unsorted_falls_back(force_numba):
    # Caller violates the "sample_indices is sorted" invariant; helper must
    # still return a globally lexsorted vector via the fallback.
    sample_indices = np.array([200, 100, 300, 100], dtype=np.int64)
    unit_indices = np.array([0, 1, 2, 0], dtype=np.int64)
    out = build_spike_vector_from_sorted_arrays(sample_indices, unit_indices)
    ref = _reference_spike_vector(sample_indices, unit_indices)
    assert np.array_equal(out, ref)


def _make_spike_vector(samples, units, segments=None):
    """Build a minimum_spike_dtype array from parallel arrays. Test helper."""
    n = len(samples)
    sv = np.empty(n, dtype=minimum_spike_dtype)
    sv["sample_index"] = samples
    sv["unit_index"] = units
    sv["segment_index"] = segments if segments is not None else 0
    return sv


def test_filter_and_remap_keep_all(force_numba):
    # Identity mapping: every parent unit_index maps to itself.
    sv = _make_spike_vector([10, 20, 30, 40], [0, 1, 2, 0])
    mapping = np.arange(3, dtype=np.int64)
    out = filter_and_remap_spike_vector(sv, mapping)
    assert np.array_equal(out, sv)


def test_filter_and_remap_drop_some(force_numba):
    # Drop unit 1 entirely; keep 0 and 2 with new indices [0, 1].
    sv = _make_spike_vector([10, 20, 30, 40, 50], [0, 1, 2, 0, 1])
    mapping = np.array([0, -1, 1], dtype=np.int64)
    out = filter_and_remap_spike_vector(sv, mapping)
    expected = _make_spike_vector([10, 30, 40], [0, 1, 0])
    assert np.array_equal(out, expected)


def test_filter_and_remap_renamed_only(force_numba):
    # Selection is full but unit indices are permuted: 0->2, 1->0, 2->1.
    sv = _make_spike_vector([10, 20, 30], [0, 1, 2])
    mapping = np.array([2, 0, 1], dtype=np.int64)
    out = filter_and_remap_spike_vector(sv, mapping)
    expected = _make_spike_vector([10, 20, 30], [2, 0, 1])
    assert np.array_equal(out, expected)


def test_filter_and_remap_empty_selection(force_numba):
    sv = _make_spike_vector([10, 20, 30], [0, 1, 2])
    mapping = np.full(3, -1, dtype=np.int64)
    out = filter_and_remap_spike_vector(sv, mapping)
    assert out.size == 0
    assert out.dtype == np.dtype(minimum_spike_dtype)


def test_filter_and_remap_empty_input(force_numba):
    sv = np.empty(0, dtype=minimum_spike_dtype)
    mapping = np.array([0, 1, 2], dtype=np.int64)
    out = filter_and_remap_spike_vector(sv, mapping)
    assert out.size == 0
    assert out.dtype == np.dtype(minimum_spike_dtype)


def test_filter_and_remap_preserves_tie_order(force_numba):
    # Two cotemporal spikes at sample 100 (units 1 and 2). After dropping unit 0,
    # the two cotemporals must appear in their original relative order — the kernel
    # never reorders within ties.
    sv = _make_spike_vector(
        [50, 100, 100, 200],
        [0, 1, 2, 1],
    )
    mapping = np.array([-1, 0, 1], dtype=np.int64)
    out = filter_and_remap_spike_vector(sv, mapping)
    expected = _make_spike_vector([100, 100, 200], [0, 1, 0])
    assert np.array_equal(out, expected)


def test_filter_and_remap_segment_index_preserved(force_numba):
    sv = _make_spike_vector([10, 20, 30, 40], [0, 1, 0, 1], segments=[0, 0, 1, 1])
    mapping = np.array([0, 1], dtype=np.int64)
    out = filter_and_remap_spike_vector(sv, mapping)
    assert np.array_equal(out["segment_index"], [0, 0, 1, 1])
    assert np.array_equal(out["sample_index"], [10, 20, 30, 40])
    assert np.array_equal(out["unit_index"], [0, 1, 0, 1])


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
