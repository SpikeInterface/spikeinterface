import pytest
import numpy as np

from spikeinterface.core import NumpySorting, UnitsSelectionSorting
from spikeinterface.core.base import minimum_spike_dtype
from spikeinterface.core.testing import check_sortings_equal

from spikeinterface.core.generate import generate_sorting


def test_basic_functions():
    sorting = generate_sorting(num_units=3, durations=[0.100, 0.100], sampling_frequency=30000.0)

    sorting2 = UnitsSelectionSorting(sorting, unit_ids=["0", "2"])
    assert np.array_equal(sorting2.unit_ids, ["0", "2"])
    assert sorting2.get_parent() == sorting

    sorting3 = UnitsSelectionSorting(sorting, unit_ids=["0", "2"], renamed_unit_ids=["a", "b"])
    assert np.array_equal(sorting3.unit_ids, ["a", "b"])

    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id="0", segment_index=0),
        sorting2.get_unit_spike_train(unit_id="0", segment_index=0),
    )
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id="0", segment_index=0),
        sorting3.get_unit_spike_train(unit_id="a", segment_index=0),
    )

    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id="2", segment_index=0),
        sorting2.get_unit_spike_train(unit_id="2", segment_index=0),
    )
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id="2", segment_index=0),
        sorting3.get_unit_spike_train(unit_id="b", segment_index=0),
    )


def test_failure_with_non_unique_unit_ids():
    seed = 10
    sorting = generate_sorting(num_units=3, durations=[0.100], sampling_frequency=30000.0, seed=seed)
    with pytest.raises(AssertionError):
        sorting2 = UnitsSelectionSorting(sorting, unit_ids=["0", "2"], renamed_unit_ids=["a", "a"])


def test_compute_and_cache_spike_vector():
    sorting = generate_sorting(num_units=3, durations=[0.100, 0.100], sampling_frequency=30000.0)

    sub_sorting = UnitsSelectionSorting(sorting, unit_ids=["2", "0"], renamed_unit_ids=["b", "a"])
    cached_spike_vector = sub_sorting.to_spike_vector(use_cache=True)
    computed_spike_vector = sub_sorting.to_spike_vector(use_cache=False)
    assert np.all(cached_spike_vector == computed_spike_vector)


PARENT_UNIT_IDS = ["22", "45", "29", "7", "3"]


def _make_parent_with_shuffled_ties(unit_ids=PARENT_UNIT_IDS, num_segments=2, num_spikes=2000, seed=42, dtype=None):
    """A sorting whose cotemporal spikes are in arbitrary unit_index order (see #4606), with
    unit_ids that are deliberately not sorted."""
    rng = np.random.default_rng(seed)
    num_units = len(unit_ids)

    # Far fewer samples than spikes, so cotemporal spikes are abundant.
    spikes = np.empty(num_spikes, dtype=minimum_spike_dtype if dtype is None else dtype)
    spikes["sample_index"] = rng.integers(0, 200, size=num_spikes)
    spikes["unit_index"] = rng.integers(0, num_units, size=num_spikes)
    spikes["segment_index"] = rng.integers(0, num_segments, size=num_spikes)
    spikes = spikes[np.lexsort((rng.random(num_spikes), spikes["sample_index"], spikes["segment_index"]))]

    sorting = NumpySorting(spikes, 30_000.0, np.asarray(unit_ids))
    assert sorting.get_num_segments() == num_segments
    return sorting


def _mask_and_remap(parent, selected_parent_ids):
    """The parent's spike vector filtered to the selected units,
    with unit_index remapped to the selection order.
    (This is the same mask the SortingAnalyzer
    extensions apply to their per-spike data.)"""
    spikes = parent.to_spike_vector()
    lut = np.full(parent.get_num_units(), -1, dtype=np.int64)
    lut[parent.ids_to_indices(selected_parent_ids)] = np.arange(len(selected_parent_ids))
    new_unit_index = lut[spikes["unit_index"]]
    keep = new_unit_index >= 0
    expected = spikes[keep].copy()
    expected["unit_index"] = new_unit_index[keep]
    return expected


def _assert_partial_invariant(spikes):
    assert np.all(np.diff(spikes["segment_index"]) >= 0)
    for segment_index in np.unique(spikes["segment_index"]):
        assert np.all(np.diff(spikes["sample_index"][spikes["segment_index"] == segment_index]) >= 0)


def _assert_has_shuffled_ties(spikes):
    full_lexsort = np.lexsort((spikes["unit_index"], spikes["sample_index"], spikes["segment_index"]))
    assert not np.array_equal(spikes, spikes[full_lexsort])


@pytest.mark.parametrize(
    "unit_ids, renamed_unit_ids",
    [
        (["29", "22", "3"], None),
        (["3", "7", "29", "45", "22"], None),
        (["22", "45"], ["b", "a"]),
        (["29", "45"], None),
    ],
    ids=["reorder", "reverse", "renamed_order_preserving", "unsorted_parent_order_preserving"],
)
def test_selection_preserves_parent_order(unit_ids, renamed_unit_ids):
    """A selection is the parent's spike vector filtered and remapped, nothing more: the parent's
    (unspecified) order of cotemporal spikes must carry over untouched."""
    parent = _make_parent_with_shuffled_ties()
    child = UnitsSelectionSorting(parent, unit_ids=unit_ids, renamed_unit_ids=renamed_unit_ids)

    expected = _mask_and_remap(parent, unit_ids)
    _assert_has_shuffled_ties(expected)

    spikes = child.to_spike_vector()
    assert np.array_equal(spikes, expected)
    _assert_partial_invariant(spikes)

    spike_trains = []
    for segment_index in range(parent.get_num_segments()):
        spike_trains.append({})
        for new_id, parent_id in zip(child.unit_ids, unit_ids):
            parent_train = parent.get_unit_spike_train(parent_id, segment_index=segment_index, use_cache=False)
            assert np.array_equal(child.get_unit_spike_train(new_id, segment_index=segment_index), parent_train)
            spike_trains[segment_index][new_id] = parent_train

    parent_counts = parent.count_num_spikes_per_unit()
    child_counts = child.count_num_spikes_per_unit()
    for new_id, parent_id in zip(child.unit_ids, unit_ids):
        assert child_counts[new_id] == parent_counts[parent_id]

    reference = NumpySorting.from_unit_dict(spike_trains, parent.sampling_frequency)
    check_sortings_equal(child, reference, check_exact_lexsort=False)


def test_selection_keeps_extra_fields():
    """Make sure fields beyond `minimum_spike_dtype`
    (e.g. the "channel_index" that `to_spike_vector(main_channel_indices=...)` adds)
    stay with their spike through a selection."""
    wide_dtype = minimum_spike_dtype + [("channel_index", "int64")]
    parent = _make_parent_with_shuffled_ties(dtype=wide_dtype)
    parent._cached_spike_vector["channel_index"] = np.arange(parent._cached_spike_vector.size)

    unit_ids = ["3", "22", "29"]
    child = UnitsSelectionSorting(parent, unit_ids=unit_ids)
    spikes = child.to_spike_vector()

    expected = _mask_and_remap(parent, unit_ids)
    assert spikes.dtype == wide_dtype
    assert np.array_equal(spikes, expected)
    assert np.array_equal(spikes["channel_index"], expected["channel_index"])


def test_zero_units_and_zero_spikes():
    parent = _make_parent_with_shuffled_ties()
    child = parent.select_units([])
    assert child.get_num_units() == 0
    assert child.to_spike_vector().size == 0
    assert child.to_spike_vector().dtype == minimum_spike_dtype
    assert child.count_num_spikes_per_unit() == {}
    assert len(child.to_spike_vector(concatenated=False)) == parent.get_num_segments()

    empty_parent = NumpySorting(np.zeros(0, dtype=minimum_spike_dtype), 30_000.0, np.array([1, 2, 3]))
    child = empty_parent.select_units([3, 1])
    assert child.to_spike_vector().size == 0
    assert np.array_equal(child._get_spike_vector_segment_slices(), [[0, 0]])


if __name__ == "__main__":
    test_basic_functions()
