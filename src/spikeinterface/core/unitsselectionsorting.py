from __future__ import annotations

import numpy as np

from .basesorting import BaseSorting, BaseSortingSegment


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

        for parent_segment in self._parent_sorting._sorting_segments:
            sub_segment = UnitsSelectionSortingSegment(parent_segment, ids_conversion)
            self.add_sorting_segment(sub_segment)

        parent_sorting.copy_metadata(self, only_main=False, ids=self._unit_ids)
        self._parent = parent_sorting

        if parent_sorting.has_recording():
            self.register_recording(parent_sorting._recording)

        self._kwargs = dict(parent_sorting=parent_sorting, unit_ids=unit_ids, renamed_unit_ids=renamed_unit_ids)

    def _custom_cache_spike_vector(self) -> None:
        if self._parent_sorting._cached_spike_vector is None:
            self._parent_sorting._custom_cache_spike_vector()

            if self._parent_sorting._cached_spike_vector is None:
                return

        parent_spike_vector = self._parent_sorting._cached_spike_vector
        parent_unit_indices = self._parent_sorting.ids_to_indices(self._unit_ids)
        sort_indices = np.argsort(parent_unit_indices)
        mask = np.isin(parent_spike_vector["unit_index"], parent_unit_indices)
        spike_vector = np.array(
            parent_spike_vector[mask]
        )  # np.array() necessary to fix 'read-only' crash with memmaps.
        indices = np.searchsorted(
            parent_unit_indices, spike_vector["unit_index"], sorter=sort_indices
        )  # Trick to make sure that the new indices are correct.
        spike_vector["unit_index"] = np.arange(len(parent_unit_indices))[sort_indices][indices]

        self._cached_spike_vector = spike_vector


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
