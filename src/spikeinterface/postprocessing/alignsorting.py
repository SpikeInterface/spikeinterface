from __future__ import annotations

import numpy as np
from typing import Optional

from spikeinterface import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.core.template_tools import get_template_extremum_channel_peak_shift


class AlignSortingExtractor(BaseSorting):
    """
    Class to shift a unit (generally to align the template on the peak) given
    the shifts for each unit.

    Parameters
    ----------
    sorting: BaseSorting
        The sorting to align.
    unit_peak_shifts: dict
        Dictionary mapping the unit_id to the unit's shift (in number of samples).
        A positive shift means the spike train is shifted back in time, while
        a negative shift means the spike train is shifted forward.

    Returns
    -------
    aligned_sorting: AlignSortingExtractor
        The aligned sorting.
    """

    def __init__(self, sorting, unit_peak_shifts):
        super().__init__(sorting.get_sampling_frequency(), sorting.unit_ids)

        for segment in sorting._sorting_segments:
            self.add_sorting_segment(AlignSortingSegment(segment, unit_peak_shifts))

        sorting.copy_metadata(self, only_main=False)
        self._parent = sorting
        if sorting.has_recording():
            self.register_recording(sorting._recording)

        self._kwargs = {"sorting": sorting, "unit_peak_shifts": unit_peak_shifts}


class AlignSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment, unit_peak_shifts):
        super().__init__()
        self._parent_segment = parent_segment
        self._unit_peak_shifts = unit_peak_shifts

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        if start_frame is not None:
            start_frame = start_frame + self._unit_peak_shifts[unit_id]
        if end_frame is not None:
            end_frame = end_frame + self._unit_peak_shifts[unit_id]
        original_spike_train = self._parent_segment.get_unit_spike_train(unit_id, start_frame, end_frame)
        return original_spike_train - self._unit_peak_shifts[unit_id]


align_sorting = define_function_from_class(source_class=AlignSortingExtractor, name="align_sorting")
