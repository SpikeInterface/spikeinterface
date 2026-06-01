import pytest
import numpy as np

from spikeinterface.extractors.phykilosortextractors import PhySortingSegment
import spikeinterface.extractors.phykilosortextractors as phymod

# Sorted spike times with known cluster assignments.
# 3 units (ids 10, 20, 30), some co-temporal spikes.
ALL_SPIKES = np.array([100, 100, 200, 300, 300, 300, 400, 500], dtype=np.int64)
ALL_CLUSTERS = np.array([10, 20, 30, 10, 20, 30, 10, 20], dtype=np.int64)
UNIT_IDS = [10, 20, 30]


@pytest.mark.parametrize("force_numpy_fallback", [False, True])
def test_phy_sorting_segment_get_unit_spike_trains(monkeypatch, force_numpy_fallback):
    """get_unit_spike_trains must match per-unit calls, for both Numba and NumPy paths."""
    if force_numpy_fallback:
        monkeypatch.setattr(phymod, "HAVE_NUMBA", False)

    seg = PhySortingSegment(ALL_SPIKES, ALL_CLUSTERS)

    # Full range, all units
    batch = seg.get_unit_spike_trains(UNIT_IDS, start_frame=None, end_frame=None)
    assert set(batch.keys()) == set(UNIT_IDS)
    for uid in UNIT_IDS:
        single = seg.get_unit_spike_train(uid, start_frame=None, end_frame=None)
        assert np.array_equal(batch[uid], single), f"Mismatch for unit {uid}"

    assert np.array_equal(batch[10], [100, 300, 400])
    assert np.array_equal(batch[20], [100, 300, 500])
    assert np.array_equal(batch[30], [200, 300])

    # With start_frame / end_frame slicing
    batch_sliced = seg.get_unit_spike_trains(UNIT_IDS, start_frame=200, end_frame=400)
    assert np.array_equal(batch_sliced[10], [300])
    assert np.array_equal(batch_sliced[20], [300])
    assert np.array_equal(batch_sliced[30], [200, 300])

    # Subset of unit_ids
    batch_subset = seg.get_unit_spike_trains([20], start_frame=None, end_frame=None)
    assert list(batch_subset.keys()) == [20]
    assert np.array_equal(batch_subset[20], [100, 300, 500])

    # Empty unit_ids
    assert seg.get_unit_spike_trains([], start_frame=None, end_frame=None) == {}


def _make_phy_folder(tmp_path):
    """Create a minimal Phy output folder for testing."""
    spike_times = np.array([100, 100, 200, 300, 300, 300, 400, 500], dtype=np.int64)
    spike_clusters = np.array([10, 20, 30, 10, 20, 30, 10, 20], dtype=np.int64)

    np.save(tmp_path / "spike_times.npy", spike_times)
    np.save(tmp_path / "spike_clusters.npy", spike_clusters)
    (tmp_path / "params.py").write_text("sample_rate = 30000.0\n")
    return tmp_path


def test_phy_compute_and_cache_spike_vector(tmp_path):
    """Phy override of _compute_and_cache_spike_vector must produce the same
    spike vector as the base class (per-unit) implementation."""
    from spikeinterface.core.basesorting import BaseSorting
    from spikeinterface.extractors.phykilosortextractors import BasePhyKilosortSortingExtractor

    phy_folder = _make_phy_folder(tmp_path)
    sorting = BasePhyKilosortSortingExtractor(phy_folder)

    # Phy override path
    sorting._compute_and_cache_spike_vector()
    phy_vector = sorting._cached_spike_vector.copy()

    # Base class (per-unit) path
    sorting._cached_spike_vector = None
    sorting._cached_spike_vector_segment_slices = None
    BaseSorting._compute_and_cache_spike_vector(sorting)
    base_vector = sorting._cached_spike_vector

    assert np.array_equal(phy_vector, base_vector)
