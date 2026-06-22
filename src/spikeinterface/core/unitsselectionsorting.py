import numpy as np

from .basesorting import BaseSorting, BaseSortingSegment
from .sorting_tools import filter_and_remap_spike_vector, is_spike_vector_sorted


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
        if self._parent_sorting._cached_spike_vector is None:
            self._parent_sorting._compute_and_cache_spike_vector()

            if self._parent_sorting._cached_spike_vector is None:
                return

        parent_unit_ids = self._parent_sorting.unit_ids

        # If the user requested an "identity selection" (all parent units, in
        # parent order, possibly renamed), the cached parent spike vector is
        # identical to the one we want — share the reference and skip the rest.
        # See `_is_identity_selection` for the definition.
        if self._is_identity_selection():
            self._cached_spike_vector = self._parent_sorting._cached_spike_vector
            parent_slices = self._parent_sorting._cached_spike_vector_segment_slices
            if parent_slices is not None:
                self._cached_spike_vector_segment_slices = parent_slices
            return

        # Build a dense LUT from parent unit_index -> new unit_index (-1 = drop).
        parent_id_to_pos = {uid: i for i, uid in enumerate(parent_unit_ids)}
        unit_mapping = np.full(parent_unit_ids.size, -1, dtype=np.int64)
        for new_idx, uid in enumerate(self._unit_ids):
            unit_mapping[parent_id_to_pos[uid]] = new_idx

        spike_vector = filter_and_remap_spike_vector(
            spike_vector=self._parent_sorting._cached_spike_vector,
            unit_mapping=unit_mapping,
        )

        # The parent's spike vector is sorted by (segment_index, sample_index, unit_index).
        # Filtering preserves that order and the remap only changes unit_index values.
        # The result stays sorted iff the selected unit_ids appear in the same relative
        # order as in the parent (an O(k) check). If not, the vector may still happen to
        # be sorted -- verify with an O(n) scan before falling back to O(n log n) lexsort.
        assume_single_segment = self.get_num_segments() == 1
        if not self._is_order_preserving_selection() and not is_spike_vector_sorted(
            spike_vector, assume_single_segment=assume_single_segment
        ):
            if assume_single_segment:
                sort_indices = np.lexsort((spike_vector["unit_index"], spike_vector["sample_index"]))
            else:
                sort_indices = np.lexsort(
                    (spike_vector["unit_index"], spike_vector["sample_index"], spike_vector["segment_index"])
                )
            spike_vector = spike_vector[sort_indices]

        self._cached_spike_vector = spike_vector

    def _is_identity_selection(self) -> bool:
        """Return True if self._unit_ids are exactly the parent's unit_ids, in parent order.

        Renaming via ``renamed_unit_ids`` does not affect this — the spike vector
        carries unit *indices*, not ids. When True, every cached form of the
        parent's spike vector (canonical, lexsorted, etc.) can be shared with
        ``self`` by reference.
        """
        parent_unit_ids = self._parent_sorting.unit_ids
        return self._unit_ids.size == parent_unit_ids.size and np.array_equal(self._unit_ids, parent_unit_ids)

    def to_reordered_spike_vector(
        self, lexsort=("sample_index", "segment_index", "unit_index"), return_order=True, return_slices=True
    ):
        # On an identity selection, the parent's lexsorted cache is exactly
        # what we'd compute — just reference it so we don't re-run the counting sort!
        if self._is_identity_selection():
            key = str(tuple(lexsort))
            if key not in self._cached_lexsorted_spike_vector:
                # Force the parent to populate its own cache (a no-op if already
                # cached) before we share the entry.
                self._parent_sorting.to_reordered_spike_vector(lexsort=lexsort, return_order=True, return_slices=True)
                parent_entry = self._parent_sorting._cached_lexsorted_spike_vector.get(key)
                if parent_entry is not None:
                    self._cached_lexsorted_spike_vector[key] = parent_entry
        return super().to_reordered_spike_vector(
            lexsort=lexsort, return_order=return_order, return_slices=return_slices
        )

    def _is_order_preserving_selection(self) -> bool:
        """Return True if self._unit_ids appear in the same relative order as in the parent.

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
