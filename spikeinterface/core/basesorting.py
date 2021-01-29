from typing import List, Union
from .mytypes import UnitId, ChannelId, SampleIndex,  ChannelIndex, Order, SamplingFrequencyHz

import numpy as np

from .base import BaseExtractor, BaseSegment




class BaseSorting(BaseExtractor):
    """
    Abstract class representing several segment several units and relative spiketrains.
    """
    def __init__(self, sampling_frequency: SamplingFrequencyHz, unit_ids: List[UnitId]):
        
        BaseExtractor.__init__(self, unit_ids)
        self._sampling_frequency = sampling_frequency
        self._sorting_segments: List[SortingSegment] = []

    def __repr__(self):
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nunits = self.get_num_units()
        sf_khz = self.get_sampling_frequency()
        txt = f'{clsname}: {nunits} nunits - {nseg} segments - {sf_khz:0.1f}kHz'
        if 'file_path' in self._kwargs:
            txt += '\n  file_path: {}'.format(self._kwargs['file_path'])
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

    def get_sampling_frequency(self):
        return self._sampling_frequency
    
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
    
    def _save_data(self, folder, format='npz', **cache_kargs):
        """
        This replace the old CacheSortingExtractor but enable more engine 
        for caching a results. At the moment only npz.
        """
        if format == 'npz':
            assert len(cache_kargs) == 0, 'Sorting.cache() with npz do not support options'
            
            from .npzsortingextractor import NpzSortingExtractor
            save_path = folder / 'sorting_cached.npz'
            NpzSortingExtractor.write_sorting(self, save_path)
            cached = NpzSortingExtractor(save_path)
        else:
            raise ValueError(f'format {format} not supported')
        
        return cached


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
