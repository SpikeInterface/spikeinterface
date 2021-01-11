from typing import List, Union

from .mytypes import ChannelId, ChannelIndex, Order, SamplingFrequencyHz, UnitId



class Sorting(object):
    """
    Abstract class representing several segment several units and relative spiketrains.
    """
    def __init__(self, sampling_frequency: SamplingFrequencyHz, unit_ids: List[UnitId]):
        self._sorting_segments: List[SortingSegment] = []
        self._unit_ids = unit_ids

    def add_sorting_segment(self, sorting_segment: SortingSegment):
        # todo: check consistency with unit ids and freq
        self._sorting_segments.append(sorting_segment)

    def get_unit_ids(self) -> List[UnitId]:
        return self._unit_ids
    
    def get_num_units(self) -> int:
        return len(self.get_unit_ids())

    def get_num_segments(self):
        return len(self._sorting_segments)
    
    def _check_segment_index(self, segment_index: Union[int, None]) -> int:
        if segment_index is None:
            if self.get_num_segments() == 1:
                return 0
            else:
                raise ValueError()
        else:
            return segment_index

    def get_unit_spike_train(self,
            unit_id: UnitId,
            segment_index: Union[int, None]=None
        ):
        segment_index = self._check_segment_index(segment_index)
        S = self._sorting_segments[segment_index]
        return S.get_unit_spike_train(unit_id=unit_id)

class SortingSegment(object):
    """
    Abstract class representing several units and relative spiketrain inside a segment.
    """
    def __init__(self):
        pass

    def get_unit_ids(self) -> List[UnitId]:
        raise NotImplementedError

    def get_sampling_frequency(self) -> SamplingFrequencyHz:
        raise NotImplementedError

    def get_unit_spike_train(self, unit_id: UnitId) -> np.ndarray:
        raise NotImplementedError
