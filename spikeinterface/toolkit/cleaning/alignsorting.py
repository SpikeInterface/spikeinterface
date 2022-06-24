import numpy as np
from typing import Optional

from spikeinterface import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class

from ..postprocessing import get_template_extremum_channel_peak_shift


class AlignSortingExtractor(BaseSorting):
    
    is_dumpable = False
    
    def __init__(self, waveform_extractor):
        sorting = waveform_extractor.sorting
        super().__init__(sorting.get_sampling_frequency(), sorting.unit_ids)
        
        unit_peak_shifts = get_template_extremum_channel_peak_shift(waveform_extractor)
        
        for segment in sorting._sorting_segments:
            self.add_sorting_segment(AlignSortingSegment(segment, unit_peak_shifts))
        
        
class AlignSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment, unit_peak_shifts):
        super().__init__()
        self._parent_segment = parent_segment
        self._unit_peak_shifts = unit_peak_shifts
        
    def get_unit_spike_train(self, unit_id, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> np.ndarray:
        original_spike_train =  self._parent_segment.get_unit_spike_train(unit_id, start_frame, end_frame)
        return original_spike_train + self._unit_peak_shifts[unit_id]


align_sorting = define_function_from_class(source_class=AlignSortingExtractor, name="align_sorting")
