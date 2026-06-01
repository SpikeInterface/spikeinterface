import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core import UnitsSelectionSorting
from spikeinterface.core.numpyextractors import NumpySorting

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

    # Spike vector must be sorted by (segment_index, sample_index, unit_index)
    n = len(spike_vector)
    if n > 1:
        seg = spike_vector["segment_index"]
        samp = spike_vector["sample_index"]
        unit = spike_vector["unit_index"]
        d_seg = np.diff(seg)
        assert np.all(d_seg >= 0), "segment_index not non-decreasing"
        seg_eq = d_seg == 0
        d_samp = np.diff(samp)
        assert np.all(d_samp[seg_eq] >= 0), "sample_index not non-decreasing within segment"
        samp_eq = seg_eq & (d_samp == 0)
        assert np.all(np.diff(unit)[samp_eq] >= 0), "unit_index not non-decreasing within same sample"


if __name__ == "__main__":
    test_basic_functions()
