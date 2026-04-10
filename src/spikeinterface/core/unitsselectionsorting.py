import numpy as np

from .basesorting import BaseSorting, BaseSortingSegment


def _is_spike_vector_sorted(spike_vector: np.ndarray) -> bool:
    """Return True iff the spike vector is sorted by (segment_index, sample_index, unit_index).

    O(n) sequential scan. Used to avoid an O(n log n) lexsort when the vector already
    happens to be in canonical order.
    """
    n = len(spike_vector)
    if n <= 1:
        return True
    seg = spike_vector["segment_index"]
    samp = spike_vector["sample_index"]
    unit = spike_vector["unit_index"]
    d_seg = np.diff(seg)
    if np.any(d_seg < 0):
        return False
    seg_eq = d_seg == 0
    d_samp = np.diff(samp)
    if np.any(d_samp[seg_eq] < 0):
        return False
    samp_eq = seg_eq & (d_samp == 0)
    if np.any(np.diff(unit)[samp_eq] < 0):
        return False
    return True


class UnitsSelectionSorting(BaseSorting):
    """
    Class that handles slicing of a Sorting object based on a list of unit_ids.

    Do not use this class directly but use `sorting.select_units(...)`

    """

    def __init__(self, parent_sorting, unit_ids=None, renamed_unit_ids=None):
        if unit_ids is None:
            unit_ids = parent_sorting.get_unit_ids()
        if renamed_unit_ids is None:
            renamed_unit_ids = unit_ids
        assert len(renamed_unit_ids) == len(np.unique(renamed_unit_ids)), "renamed_unit_ids must be unique!"

        self._parent_sorting = parent_sorting
        self._unit_ids = np.asarray(unit_ids)
        self._renamed_unit_ids = np.asarray(renamed_unit_ids)

        parents_unit_ids = parent_sorting.get_unit_ids()
        sampling_frequency = parent_sorting.get_sampling_frequency()

        # some checks
        assert all(unit_id in parents_unit_ids for unit_id in self._unit_ids), "unit ids are not all in parents"
        assert len(self._unit_ids) == len(self._renamed_unit_ids), "renamed channel_ids must be the same size"

        ids_conversion = dict(zip(self._renamed_unit_ids, self._unit_ids))

        BaseSorting.__init__(self, sampling_frequency, self._renamed_unit_ids)

        for parent_segment in self._parent_sorting.segments:
            sub_segment = UnitsSelectionSortingSegment(parent_segment, ids_conversion)
            self.add_sorting_segment(sub_segment)

        parent_sorting.copy_metadata(self, only_main=False, ids=self._unit_ids)
        self._parent = parent_sorting

        if parent_sorting.has_recording():
            self.register_recording(parent_sorting._recording)

        self._kwargs = dict(parent_sorting=parent_sorting, unit_ids=unit_ids, renamed_unit_ids=renamed_unit_ids)

    def _compute_and_cache_spike_vector(self) -> None:
        from spikeinterface.core.sorting_tools import remap_unit_indices_in_vector

        if self._parent_sorting._cached_spike_vector is None:
            self._parent_sorting._compute_and_cache_spike_vector()

            if self._parent_sorting._cached_spike_vector is None:
                return

        spike_vector, _ = remap_unit_indices_in_vector(
            vector=self._parent_sorting._cached_spike_vector,
            all_old_unit_ids=self._parent_sorting.unit_ids,
            all_new_unit_ids=self._unit_ids,
        )

        # The parent's spike vector is sorted by (segment_index, sample_index, unit_index).
        # Boolean filtering by unit preserves that order; the remap only changes unit_index
        # values. The result stays sorted iff the selected unit_ids appear in the same
        # relative order as in the parent (an O(k) check). If not, the vector may still
        # happen to be sorted -- verify with an O(n) scan before falling back to O(n log n)
        # lexsort.
        if not self._is_order_preserving_selection() and not _is_spike_vector_sorted(spike_vector):
            sort_indices = np.lexsort(
                (spike_vector["unit_index"], spike_vector["sample_index"], spike_vector["segment_index"])
            )
            spike_vector = spike_vector[sort_indices]

        self._cached_spike_vector = spike_vector

    def _is_order_preserving_selection(self) -> bool:
        """Return True iff self._unit_ids appear in the same relative order as in the parent.

        O(k) where k is the number of selected units. When True, the remapped spike vector
        is guaranteed to remain sorted by (segment, sample, unit) without re-sorting.
        """
        parent_unit_ids = self._parent_sorting.unit_ids
        parent_id_to_pos = {uid: i for i, uid in enumerate(parent_unit_ids)}
        prev_pos = -1
        for uid in self._unit_ids:
            pos = parent_id_to_pos.get(uid)
            if pos is None or pos <= prev_pos:
                return False
            prev_pos = pos
        return True


class UnitsSelectionSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment, ids_conversion):
        BaseSortingSegment.__init__(self)
        self._parent_segment = parent_segment
        self._ids_conversion = ids_conversion

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> np.ndarray:
        unit_id_parent = self._ids_conversion[unit_id]
        times = self._parent_segment.get_unit_spike_train(unit_id_parent, start_frame, end_frame)
        return times

    def get_unit_spike_trains(
        self,
        unit_ids,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> dict:
        unit_ids_parent = [self._ids_conversion[unit_id] for unit_id in unit_ids]
        parent_trains = self._parent_segment.get_unit_spike_trains(unit_ids_parent, start_frame, end_frame)
        return {child_id: parent_trains[parent_id] for child_id, parent_id in zip(unit_ids, unit_ids_parent)}
