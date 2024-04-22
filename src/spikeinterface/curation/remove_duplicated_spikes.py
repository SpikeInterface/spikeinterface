from __future__ import annotations
from typing import Optional
import numpy as np
from spikeinterface.core import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.curation.curation_tools import find_duplicated_spikes


class RemoveDuplicatedSpikesSorting(BaseSorting):
    """
    Class to remove duplicated spikes from the spike trains.
    Spikes are considered duplicated if they are less than x
    ms apart where x is the censored period.

    Parameters
    ----------
    sorting: BaseSorting
        The parent sorting.
    censored_period_ms: float
        The censored period to consider 2 spikes to be duplicated (in ms).
    method: "keep_first" | "keep_last" | "keep_first_iterative" | "keep_last_iterative" | "random", default: "keep_first"
        Method used to remove the duplicated spikes.
        If method = "random", will randomly choose to remove the first or last spike.
        If method = "keep_first", for each ISI violation, will remove the second spike.
        If method = "keep_last", for each ISI violation, will remove the first spike.
        If method = "keep_first_iterative", will iteratively keep the first spike and remove the following violations.
        If method = "keep_last_iterative", does the same as "keep_first_iterative" but starting from the end.
        In the iterative methods, if there is a triplet A, B, C where (A, B) and (B, C) are in the censored period
        (but not (A, C)), then only B is removed. In the non iterative methods however, only one spike remains.

    Returns
    -------
    sorting_without_duplicated_spikes: Remove_DuplicatedSpikesSorting
        The sorting without any duplicated spikes.
    """

    def __init__(self, sorting: BaseSorting, censored_period_ms: float = 0.3, method: str = "keep_first") -> None:
        super().__init__(sorting.get_sampling_frequency(), sorting.unit_ids)
        censored_period = int(round(censored_period_ms * 1e-3 * sorting.get_sampling_frequency()))
        seed = np.random.randint(low=0, high=np.iinfo(np.int32).max)

        for segment in sorting._sorting_segments:
            self.add_sorting_segment(
                RemoveDuplicatedSpikesSortingSegment(segment, censored_period, sorting.unit_ids, method, seed)
            )

        sorting.copy_metadata(self, only_main=False)
        self._parent = sorting
        if sorting.has_recording():
            self.register_recording(sorting._recording)

        self._kwargs = {"sorting": sorting, "censored_period_ms": censored_period_ms, "method": method}


class RemoveDuplicatedSpikesSortingSegment(BaseSortingSegment):
    def __init__(
        self,
        parent_segment: BaseSortingSegment,
        censored_period: int,
        unit_ids,
        method: str,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._parent_segment = parent_segment
        self.censored_period = censored_period
        self.method = method
        self.seed = seed
        self._duplicated_spikes = {}

    def get_unit_spike_train(
        self, unit_id, start_frame: Optional[int] = None, end_frame: Optional[int] = None
    ) -> np.ndarray:
        spike_train = self._parent_segment.get_unit_spike_train(unit_id, start_frame=None, end_frame=None)

        if unit_id not in self._duplicated_spikes:
            self._duplicated_spikes[unit_id] = find_duplicated_spikes(
                spike_train,
                self.censored_period,
                method=self.method,
                seed=self.seed,
            )

        spike_train = np.delete(spike_train, self._duplicated_spikes[unit_id])

        if start_frame == None:
            start_frame = 0
        if end_frame == None:
            end_frame = spike_train[-1] if len(spike_train) > 0 else 0

        start = np.searchsorted(spike_train, start_frame, side="left")
        end = np.searchsorted(spike_train, end_frame, side="right")

        return spike_train[start:end]


remove_duplicated_spikes = define_function_from_class(
    source_class=RemoveDuplicatedSpikesSorting, name="remove_duplicated_spikes"
)
