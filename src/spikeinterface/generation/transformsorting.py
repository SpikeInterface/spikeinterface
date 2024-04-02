from __future__ import annotations

import numpy as np
from typing import Optional, List, Union

from ..core import BaseSorting, SpikeVectorSortingSegment
from ..core.numpyextractors import NumpySorting
from ..core.basesorting import minimum_spike_dtype


class TransformSorting(BaseSorting):
    """
    Generates a sorting object keeping track of added spikes/units from an external spike_vector.
    More precisely, the TransformSorting objects keeps two internal arrays added_spikes_from_existing_units and
    added_spikes_from_new_units as boolean mask to track (in the representation as a spike vector) where
    modifications have been made

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object
    added_spikes_existing_units : np.array (spike_vector)
        The spikes that should be added to the sorting object, for existing units
    added_spikes_new_units: np.array (spike_vector)
        The spikes that should be added to the sorting object, for new units
    new_units_ids: list
        The unit_ids that should be added if spikes for new units are added
    refractory_period_ms : float, default None
        The refractory period violation to prevent duplicates and/or unphysiological addition
        of spikes. Any spike times in added_spikes violating the refractory period will be
        discarded

    Returns
    -------
    sorting : TransformSorting
        The sorting object with the added spikes and/or units
    """

    def __init__(
        self,
        sorting: BaseSorting,
        added_spikes_existing_units=None,
        added_spikes_new_units=None,
        new_unit_ids: Optional[List[Union[str, int]]] = None,
        refractory_period_ms: Optional[float] = None,
    ):
        sampling_frequency = sorting.get_sampling_frequency()
        unit_ids = list(sorting.get_unit_ids())

        if new_unit_ids is not None:
            new_unit_ids = list(new_unit_ids)
            assert ~np.any(
                np.isin(new_unit_ids, sorting.unit_ids)
            ), "some units ids are already present. Consider using added_spikes_existing_units"
            if len(new_unit_ids) > 0:
                assert type(unit_ids[0]) == type(new_unit_ids[0]), "unit_ids should have the same type"
                unit_ids = unit_ids + list(new_unit_ids)

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        self.parent_unit_ids = sorting.unit_ids
        self._cached_spike_vector = sorting.to_spike_vector().copy()
        self.refractory_period_ms = refractory_period_ms

        self.added_spikes_from_existing_mask = np.zeros(len(self._cached_spike_vector), dtype=bool)
        self.added_spikes_from_new_mask = np.zeros(len(self._cached_spike_vector), dtype=bool)

        if added_spikes_existing_units is not None and len(added_spikes_existing_units) > 0:
            assert (
                added_spikes_existing_units.dtype == minimum_spike_dtype
            ), "added_spikes_existing_units should be a spike vector"
            added_unit_indices = np.arange(len(self.parent_unit_ids))
            self._cached_spike_vector = np.concatenate((self._cached_spike_vector, added_spikes_existing_units))
            self.added_spikes_from_existing_mask = np.concatenate(
                (self.added_spikes_from_existing_mask, np.ones(len(added_spikes_existing_units), dtype=bool))
            )
            self.added_spikes_from_new_mask = np.concatenate(
                (self.added_spikes_from_new_mask, np.zeros(len(added_spikes_existing_units), dtype=bool))
            )

        if added_spikes_new_units is not None and len(added_spikes_new_units) > 0:
            assert (
                added_spikes_new_units.dtype == minimum_spike_dtype
            ), "added_spikes_new_units should be a spike vector"
            self._cached_spike_vector = np.concatenate((self._cached_spike_vector, added_spikes_new_units))
            self.added_spikes_from_existing_mask = np.concatenate(
                (self.added_spikes_from_existing_mask, np.zeros(len(added_spikes_new_units), dtype=bool))
            )
            self.added_spikes_from_new_mask = np.concatenate(
                (self.added_spikes_from_new_mask, np.ones(len(added_spikes_new_units), dtype=bool))
            )

        sort_idxs = np.lexsort([self._cached_spike_vector["sample_index"], self._cached_spike_vector["segment_index"]])
        self._cached_spike_vector = self._cached_spike_vector[sort_idxs]
        self.added_spikes_from_existing_mask = self.added_spikes_from_existing_mask[sort_idxs]
        self.added_spikes_from_new_mask = self.added_spikes_from_new_mask[sort_idxs]

        # We need to add the sorting segments
        for segment_index in range(sorting.get_num_segments()):
            segment = SpikeVectorSortingSegment(self._cached_spike_vector, segment_index, unit_ids=self.unit_ids)
            self.add_sorting_segment(segment)

        if self.refractory_period_ms is not None:
            self.clean_refractory_period()

        self._kwargs = dict(
            sorting=sorting,
            added_spikes_existing_units=added_spikes_existing_units,
            added_spikes_new_units=added_spikes_new_units,
            new_unit_ids=new_unit_ids,
            refractory_period_ms=refractory_period_ms,
        )

    @property
    def added_spikes_mask(self):
        return np.logical_or(self.added_spikes_from_existing_mask, self.added_spikes_from_new_mask)

    def get_added_spikes_indices(self):
        return np.nonzero(self.added_spikes_mask)[0]

    def get_added_spikes_from_existing_indices(self):
        return np.nonzero(self.added_spikes_from_existing_mask)[0]

    def get_added_spikes_from_new_indices(self):
        return np.nonzero(self.added_spikes_from_new_mask)[0]

    def get_added_units_inds(self):
        return self.unit_ids[len(self.parent_unit_ids) :]

    @staticmethod
    def add_from_sorting(sorting1: BaseSorting, sorting2: BaseSorting, refractory_period_ms=None) -> "TransformSorting":
        """
        Construct TransformSorting by adding one sorting to one other.

        Parameters
        ----------
        sorting1: the first sorting
        sorting2: the second sorting
        refractory_period_ms : float, default None
            The refractory period violation to prevent duplicates and/or unphysiological addition
            of spikes. Any spike times in added_spikes violating the refractory period will be
            discarded
        """
        assert (
            sorting1.get_sampling_frequency() == sorting2.get_sampling_frequency()
        ), "sampling_frequency should be the same"
        assert type(sorting1.unit_ids[0]) == type(sorting2.unit_ids[0]), "unit_ids should have the same type"
        # We detect the indices that are shared by the two sortings
        mask1 = np.isin(sorting2.unit_ids, sorting1.unit_ids)
        common_ids = sorting2.unit_ids[mask1]
        exclusive_ids = sorting2.unit_ids[~mask1]

        # We detect the indicies in the spike_vectors
        idx1 = sorting1.ids_to_indices(common_ids)
        idx2 = sorting2.ids_to_indices(common_ids)

        spike_vector_2 = sorting2.to_spike_vector()
        from_existing_units = np.isin(spike_vector_2["unit_index"], idx2)
        common = spike_vector_2[from_existing_units].copy()

        # If indices are not the same, we need to remap
        if not np.all(idx1 == idx2):
            old_indices = common["unit_index"].copy()
            for i, j in zip(idx1, idx2):
                mask = old_indices == j
                common["unit_index"][mask] = i

        idx1 = len(sorting1.unit_ids) + np.arange(len(exclusive_ids), dtype=int)
        idx2 = sorting2.ids_to_indices(exclusive_ids)

        not_common = spike_vector_2[~from_existing_units].copy()

        # If indices are not the same, we need to remap
        if not np.all(idx1 == idx2):
            old_indices = not_common["unit_index"].copy()
            for i, j in zip(idx1, idx2):
                mask = old_indices == j
                not_common["unit_index"][mask] = i

        sorting = TransformSorting(
            sorting1,
            added_spikes_existing_units=common,
            added_spikes_new_units=not_common,
            new_unit_ids=exclusive_ids,
            refractory_period_ms=refractory_period_ms,
        )
        return sorting

    @staticmethod
    def add_from_unit_dict(
        sorting1: BaseSorting, units_dict_list: dict, refractory_period_ms=None
    ) -> "TransformSorting":
        """
        Construct TransformSorting by adding one sorting with a
        list of dict. The list length is the segment count.
        Each dict have unit_ids as keys and spike times as values.

        Parameters
        ----------

        sorting1: the first sorting
        dict_list: list of dict
        refractory_period_ms : float, default None
            The refractory period violation to prevent duplicates and/or unphysiological addition
            of spikes. Any spike times in added_spikes violating the refractory period will be
            discarded
        """
        sorting2 = NumpySorting.from_unit_dict(units_dict_list, sorting1.get_sampling_frequency())
        sorting = TransformSorting.add_from_sorting(sorting1, sorting2, refractory_period_ms)
        return sorting

    @staticmethod
    def from_times_labels(
        sorting1, times_list, labels_list, sampling_frequency, unit_ids=None, refractory_period_ms=None
    ) -> "NumpySorting":
        """
        Construct TransformSorting from:
          * an array of spike times (in frames)
          * an array of spike labels and adds all the
        In case of multisegment, it is a list of array.

        Parameters
        ----------
        sorting1: the first sorting
        times_list: list of array (or array)
            An array of spike times (in frames)
        labels_list: list of array (or array)
            An array of spike labels corresponding to the given times
        unit_ids: list or None, default: None
            The explicit list of unit_ids that should be extracted from labels_list
            If None, then it will be np.unique(labels_list)
        refractory_period_ms : float, default None
            The refractory period violation to prevent duplicates and/or unphysiological addition
            of spikes. Any spike times in added_spikes violating the refractory period will be
            discarded
        """

        sorting2 = NumpySorting.from_times_labels(times_list, labels_list, sampling_frequency, unit_ids)
        sorting = TransformSorting.add_from_sorting(sorting1, sorting2, refractory_period_ms)
        return sorting

    def clean_refractory_period(self):
        ## This function will remove the added spikes that will violate RPV, but does not affect the
        ## spikes in the original sorting. So if some RPV violation are present in this sorting,
        ## they will be left untouched
        unit_indices = np.unique(self._cached_spike_vector["unit_index"])
        rpv = int(self.get_sampling_frequency() * self.refractory_period_ms / 1000)
        to_keep = ~self.added_spikes_from_existing_mask.copy()
        for segment_index in range(self.get_num_segments()):
            for unit_ind in unit_indices:
                (indices,) = np.nonzero(
                    (self._cached_spike_vector["unit_index"] == unit_ind)
                    * (self._cached_spike_vector["segment_index"] == segment_index)
                )
                to_keep[indices[1:]] = np.logical_or(
                    to_keep[indices[1:]], np.diff(self._cached_spike_vector[indices]["sample_index"]) > rpv
                )

        self._cached_spike_vector = self._cached_spike_vector[to_keep]
        self.added_spikes_from_existing_mask = self.added_spikes_from_existing_mask[to_keep]
        self.added_spikes_from_new_mask = self.added_spikes_from_new_mask[to_keep]
