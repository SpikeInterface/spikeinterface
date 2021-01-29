from typing import List, Union
from .mytypes import UnitId, ChannelId, SampleIndex,  ChannelIndex, Order, SamplingFrequencyHz

import numpy as np

from .base import BaseExtractor, BaseSegment




class BaseSorting(BaseExtractor):
    """
    Abstract class representing several segment several units and relative spiketrains.
    """
    def __init__(self, sampling_frequency: SamplingFrequencyHz, unit_ids: List[UnitId]):
        
        ExtractorBase.__init__(self, unit_ids)
        self._sorting_segments: List[SortingSegment] = []

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nunits = self.get_num_units()
        sf_khz = self.get_sampling_frequency()
        txt = f'{clsname}: {nunits} nunits - {nseg} segments - {sf_khz:0.1f}kHz'
        return txt

    @property
    def unit_ids(self):
        return self._main_ids 

    def get_unit_ids(self) -> List[UnitId]:
        return  self._main_ids

    def get_num_units(self) -> int:
        return len(self.get_unit_ids())

    def add_sorting_segment(self, sorting_segment):
        # todo: check consistency with unit ids and freq
        self._sorting_segments.append(sorting_segment)
        sorting_segment.set_parent_extractor(self)
    
    def get_num_segments(self):
        return len(self._sorting_segments)
 
    def get_unit_spike_train(self,
            unit_id: UnitId,
            segment_index: Union[int, None]=None,
            start_frame: Union[SampleIndex, None]=None,
            end_frame: Union[SampleIndex, None]=None,
        ):
        segment_index = self._check_segment_index(segment_index)
        S = self._sorting_segments[segment_index]
        return S.get_unit_spike_train(unit_id=unit_id, start_frame=start_frame, end_frame=end_frame)


class BaseSortingSegment(BaseSegment):
    """
    Abstract class representing several units and relative spiketrain inside a segment.
    """
    def __init__(self):
        BaseSegment.__init__(self)
    
    def get_unit_spike_train(self, 
            unit_id: UnitId,
           start_frame: Union[SampleIndex, None] = None,
           end_frame: Union[SampleIndex, None] = None,
        ) -> np.ndarray:
        # must be implemented in subclass
        raise NotImplementedError
