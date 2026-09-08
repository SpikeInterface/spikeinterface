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
    apply_splits_to_sorting,
    _get_ids_after_merging,
    generate_unit_ids_for_merge_group,
    remap_unit_indices_in_vector,
    set_properties_after_merging,
    set_properties_after_splits,
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


@pytest.mark.parametrize("method", ["uniform", "percentage", "maximum_rate", "all"])
def test_random_spikes_selection_no_unit(method):
    # a sorting with no unit is valid and should give an empty selection, not raise
    recording, sorting = generate_ground_truth_recording(
        durations=[5.0],
        sampling_frequency=16000.0,
        num_channels=4,
        num_units=3,
        seed=2205,
    )
    empty_sorting = sorting.select_units([])
    num_samples = [recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())]

    random_spikes_indices = random_spikes_selection(
        empty_sorting, num_samples, method=method, percentage=0.5, maximum_rate=10.0, seed=2205
    )
    assert random_spikes_indices.size == 0


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


def _make_sorting_with_properties():
    """Helper: 4-unit sorting with float and str properties."""
    times = np.array([0, 10, 20, 30, 40])
    labels = np.array(["a", "b", "c", "d", "a"])
    sorting = NumpySorting.from_samples_and_labels([times], [labels], 10_000.0, unit_ids=["a", "b", "c", "d"])
    # same value for "a" and "b", different for "c" and "d"
    sorting.set_property("quality", np.array([1.0, 1.0, 2.0, 3.0]))
    sorting.set_property("group", np.array(["g1", "g1", "g2", "g3"]))
    return sorting


def test_set_properties_after_merging():
    sorting = _make_sorting_with_properties()

    # --- append strategy (baseline) ---
    sorting_merged, _, _ = apply_merges_to_sorting(
        sorting, [["a", "b"]], censor_ms=None, new_id_strategy="append", return_extra=True
    )
    is_merged = sorting_merged.get_property("is_merged")
    # "merge0" is the new merged unit; "c" and "d" are kept
    merged_idx = sorting_merged.id_to_index("merge0")
    kept_c_idx = sorting_merged.id_to_index("c")
    kept_d_idx = sorting_merged.id_to_index("d")
    assert is_merged[merged_idx] is True or bool(is_merged[merged_idx])
    assert not is_merged[kept_c_idx]
    assert not is_merged[kept_d_idx]
    # same quality value for "a" and "b" → propagated
    quality = sorting_merged.get_property("quality")
    assert quality[merged_idx] == 1.0

    # --- take_first strategy (the bug fix) ---
    sorting_merged2, _, _ = apply_merges_to_sorting(
        sorting, [["a", "b"]], censor_ms=None, new_id_strategy="take_first", return_extra=True
    )
    is_merged2 = sorting_merged2.get_property("is_merged")
    # "a" is the new merged unit (take_first); "c" and "d" are kept
    merged_idx2 = sorting_merged2.id_to_index("a")
    kept_c_idx2 = sorting_merged2.id_to_index("c")
    kept_d_idx2 = sorting_merged2.id_to_index("d")
    assert is_merged2[merged_idx2]  # was False before the fix
    assert not is_merged2[kept_c_idx2]
    assert not is_merged2[kept_d_idx2]
    # same quality for "a" and "b" → propagated (merge logic, not raw copy)
    quality2 = sorting_merged2.get_property("quality")
    assert quality2[merged_idx2] == 1.0

    # --- join strategy ---
    sorting_merged3, _, _ = apply_merges_to_sorting(
        sorting, [["a", "b"]], censor_ms=None, new_id_strategy="join", return_extra=True
    )
    is_merged3 = sorting_merged3.get_property("is_merged")
    merged_idx3 = sorting_merged3.id_to_index("a-b")
    assert is_merged3[merged_idx3]

    # --- multiple merge groups ---
    sorting5 = NumpySorting.from_samples_and_labels(
        [np.array([0, 10, 20, 30, 40, 50])],
        [np.array(["a", "b", "c", "d", "e", "f"])],
        10_000.0,
        unit_ids=["a", "b", "c", "d", "e", "f"],
    )
    sorting5.set_property("quality", np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))
    sorting_merged5 = apply_merges_to_sorting(
        sorting5, [["a", "b"], ["c", "d"]], censor_ms=None, new_id_strategy="take_first"
    )
    is_merged5 = sorting_merged5.get_property("is_merged")
    assert is_merged5[sorting_merged5.id_to_index("a")]
    assert is_merged5[sorting_merged5.id_to_index("c")]
    assert not is_merged5[sorting_merged5.id_to_index("e")]
    assert not is_merged5[sorting_merged5.id_to_index("f")]

    # --- different property values for merged units → default fill ---
    sorting_diff = NumpySorting.from_samples_and_labels(
        [np.array([0, 10, 20])],
        [np.array(["a", "b", "c"])],
        10_000.0,
        unit_ids=["a", "b", "c"],
    )
    sorting_diff.set_property("quality", np.array([1.0, 2.0, 3.0]))  # a≠b
    sorting_diff_merged = apply_merges_to_sorting(
        sorting_diff, [["a", "b"]], censor_ms=None, new_id_strategy="take_first"
    )
    is_merged_diff = sorting_diff_merged.get_property("is_merged")
    assert is_merged_diff[sorting_diff_merged.id_to_index("a")]
    assert not is_merged_diff[sorting_diff_merged.id_to_index("c")]


def test_set_properties_after_splits():
    times = np.array([0, 10, 20, 30, 40])
    labels = np.array(["a", "b", "b", "c", "c"])
    sorting = NumpySorting.from_samples_and_labels([times], [labels], 10_000.0, unit_ids=["a", "b", "c"])
    sorting.set_property("quality", np.array([1.0, 2.0, 3.0]))
    sorting.set_property("group", np.array(["g1", "g2", "g3"]))

    # --- append strategy: split "b" into two new units ---
    unit_splits = {"b": [np.array([0]), np.array([1])]}
    sorting_split, new_unit_ids = apply_splits_to_sorting(
        sorting, unit_splits, new_id_strategy="append", return_extra=True
    )
    is_split = sorting_split.get_property("is_split")
    # new units for "b" → is_split=True; "a" and "c" kept → is_split=False
    for new_uid in new_unit_ids[0]:
        assert is_split[sorting_split.id_to_index(new_uid)]
    assert not is_split[sorting_split.id_to_index("a")]
    assert not is_split[sorting_split.id_to_index("c")]
    # quality of "b" (2.0) propagated to both sub-units
    quality = sorting_split.get_property("quality")
    for new_uid in new_unit_ids[0]:
        assert quality[sorting_split.id_to_index(new_uid)] == 2.0

    # --- split strategy with str unit ids ---
    sorting_split2, new_unit_ids2 = apply_splits_to_sorting(
        sorting, unit_splits, new_id_strategy="split", return_extra=True
    )
    is_split2 = sorting_split2.get_property("is_split")
    # new units are "b-0" and "b-1"
    for new_uid in new_unit_ids2[0]:
        assert is_split2[sorting_split2.id_to_index(new_uid)]
    assert not is_split2[sorting_split2.id_to_index("a")]
    assert not is_split2[sorting_split2.id_to_index("c")]

    # --- edge case: call set_properties_after_splits directly with a new split unit id
    #     that overlaps with an existing pre-split unit id (defensive fix) ---
    # Manually build post-split sorting: "b" was split into ["a_new", "x"] but we
    # simulate a hypothetical overlap by calling set_properties_after_splits directly
    # with new_unit_ids containing a pre-existing id.
    times2 = np.array([0, 10, 20, 30])
    labels2 = np.array(["a", "x", "x", "c"])
    sorting_post = NumpySorting.from_samples_and_labels([times2], [labels2], 10_000.0, unit_ids=["a", "x", "c"])
    sorting_post.set_property("quality", np.empty(3))
    # "x" is a new split sub-unit of "b"; "a" and "c" are kept
    # In this case none of the new split unit ids are in pre_unit_ids, so untouched_unit_ids=["a","c"]
    set_properties_after_splits(
        sorting_post,
        sorting,
        split_unit_ids=["b"],
        new_unit_ids=[["x", "c"]],  # "c" overlaps with pre_unit_ids — the edge case
    )
    is_split3 = sorting_post.get_property("is_split")
    # "c" is a new split unit here (overlapping id), so it must be is_split=True
    assert is_split3[sorting_post.id_to_index("c")]
    assert is_split3[sorting_post.id_to_index("x")]
    assert not is_split3[sorting_post.id_to_index("a")]


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
