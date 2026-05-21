import pytest
import numpy as np

from spikeinterface.core import UnitsSelectionSorting
from spikeinterface.core.numpyextractors import NumpySorting
from spikeinterface.core.sorting_tools import is_spike_vector_sorted

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
        UnitsSelectionSorting(sorting, unit_ids=["0", "2"], renamed_unit_ids=["a", "a"])


def test_compute_and_cache_spike_vector():
    """USS override of _compute_and_cache_spike_vector must produce the same
    spike vector as the base class (per-unit) implementation."""
    from spikeinterface.core.basesorting import BaseSorting

    sorting = generate_sorting(num_units=3, durations=[0.100, 0.100], sampling_frequency=30000.0)

    sub_sorting = UnitsSelectionSorting(sorting, unit_ids=["2", "0"], renamed_unit_ids=["b", "a"])

    # USS override path
    sub_sorting._compute_and_cache_spike_vector()
    uss_vector = sub_sorting._cached_spike_vector.copy()

    # Base class (per-unit) path
    sub_sorting._cached_spike_vector = None
    sub_sorting._cached_spike_vector_segment_slices = None
    BaseSorting._compute_and_cache_spike_vector(sub_sorting)
    base_vector = sub_sorting._cached_spike_vector

    assert np.array_equal(uss_vector, base_vector)


@pytest.mark.parametrize("use_cache", [False, True])
def test_uss_get_unit_spike_trains_with_renamed_ids(use_cache):
    """get_unit_spike_trains on a USS with renamed ids must return dicts with child ids
    (as opposed to parent ids) as keys."""
    sorting = generate_sorting(num_units=5, durations=[0.100], sampling_frequency=30000.0, seed=42)

    # Select a subset and rename
    sub = UnitsSelectionSorting(sorting, unit_ids=["1", "3", "4"], renamed_unit_ids=["a", "b", "c"])
    renamed_ids = list(sub.unit_ids)

    batch = sub.get_unit_spike_trains(unit_ids=renamed_ids, segment_index=0, use_cache=use_cache)

    assert isinstance(batch, dict)
    assert set(batch.keys()) == set(renamed_ids)

    for uid in renamed_ids:
        single = sub.get_unit_spike_train(unit_id=uid, segment_index=0, use_cache=use_cache)
        assert np.array_equal(batch[uid], single), f"Mismatch for unit {uid}"


def test_spike_vector_sorted_after_reorder_with_cotemporal_spikes():
    """USS spike vector must be correctly sorted even when selection reverses unit order
    and co-temporal spikes exist (same sample_index, different units)."""
    from spikeinterface.core.basesorting import BaseSorting

    # Build a sorting with guaranteed co-temporal spikes:
    # units 0, 1, 2 all fire at sample 100 and 200
    samples = np.array([100, 100, 100, 200, 200, 200, 300, 400], dtype=np.int64)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)
    sorting = NumpySorting.from_samples_and_labels(
        samples_list=[samples], labels_list=[labels], sampling_frequency=30000.0
    )

    # Reverse the unit order — _is_order_preserving_selection must return False
    sub = UnitsSelectionSorting(sorting, unit_ids=[2, 0], renamed_unit_ids=["b", "a"])

    spike_vector = sub.to_spike_vector()

    sub._cached_spike_vector = None
    sub._cached_spike_vector_segment_slices = None
    BaseSorting._compute_and_cache_spike_vector(sub)
    base_vector = sub._cached_spike_vector

    assert np.array_equal(spike_vector, base_vector)
    assert np.all(spike_vector["segment_index"] == 0)
    assert is_spike_vector_sorted(spike_vector)


def test_compute_and_cache_spike_vector_identity_selection_shares_parent_cache():
    """A USS that selects all of its parent's units in parent order should reuse the
    parent's cached spike vector by reference, not rebuild it."""
    from spikeinterface.core.basesorting import BaseSorting

    sorting = generate_sorting(num_units=4, durations=[0.100, 0.100], sampling_frequency=30000.0)

    # First USS: identity selection over `sorting`. Force its cache.
    uss1 = UnitsSelectionSorting(sorting, unit_ids=list(sorting.unit_ids))
    uss1._compute_and_cache_spike_vector()
    assert uss1._cached_spike_vector is not None

    # Second USS: identity selection over uss1, with renamed ids to exercise the
    # rename-only path. The cached spike vector must be the same Python object.
    renamed = [f"r{uid}" for uid in uss1.unit_ids]
    uss2 = UnitsSelectionSorting(uss1, unit_ids=list(uss1.unit_ids), renamed_unit_ids=renamed)
    uss2._compute_and_cache_spike_vector()
    assert uss2._cached_spike_vector is uss1._cached_spike_vector
    if uss1._cached_spike_vector_segment_slices is not None:
        assert uss2._cached_spike_vector_segment_slices is uss1._cached_spike_vector_segment_slices

    # Belt-and-suspenders: the shared vector must still match the slow base-class path.
    uss2._cached_spike_vector = None
    uss2._cached_spike_vector_segment_slices = None
    BaseSorting._compute_and_cache_spike_vector(uss2)
    base_vector = uss2._cached_spike_vector
    assert np.array_equal(uss1._cached_spike_vector, base_vector)


def test_to_reordered_spike_vector_identity_selection_shares_parent_cache():
    """A USS that selects all of its parent's units in parent order should reuse the
    parent's lexsorted spike vector cache by reference, not re-run the counting sort."""
    sorting = generate_sorting(num_units=5, durations=[0.200, 0.200], sampling_frequency=30000.0)

    # Identity selection, with renamed ids to also exercise the rename-only path.
    renamed = [f"r{uid}" for uid in sorting.unit_ids]
    uss = UnitsSelectionSorting(sorting, unit_ids=list(sorting.unit_ids), renamed_unit_ids=renamed)

    for lexsort in [
        ("sample_index", "segment_index", "unit_index"),
        ("sample_index", "unit_index", "segment_index"),
    ]:
        # Force the parent to build the lexsorted cache.
        parent_ordered, _, parent_slices = sorting.to_reordered_spike_vector(
            lexsort=lexsort, return_order=True, return_slices=True
        )
        key = str(lexsort)
        assert key in sorting._cached_lexsorted_spike_vector

        # Reset USS cache and force a build through the override.
        uss._cached_lexsorted_spike_vector = {}
        uss_ordered, _, uss_slices = uss.to_reordered_spike_vector(
            lexsort=lexsort, return_order=True, return_slices=True
        )

        # The cache entry must be the *same* dict object as the parent's.
        assert (
            uss._cached_lexsorted_spike_vector[key] is sorting._cached_lexsorted_spike_vector[key]
        ), f"identity USS did not share parent lexsorted cache for {lexsort}"
        assert uss_ordered is parent_ordered
        assert uss_slices is parent_slices


if __name__ == "__main__":
    test_basic_functions()
