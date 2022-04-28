from typing import List, Union

import numpy as np
from copy import deepcopy
from .basesorting import BaseSorting, BaseSortingSegment

class SplitSortingUnit(BaseSorting):
    """
    Class that handles merging units of a Sorting object based on a list of unit_ids.

    Parameters
    ----------
    parent_sorting: Recording
        The recording object
    parent_unit_id: int
        Unit id of the unit to split        
    indices_list: list
        A list of boolean arrays selecting the spikes to split in each segmet 
    new_units_ids: int
        Units id of the new units to be created. [False flag, True flag]  
    cache: Bool

    Returns
    -------
    sorting: Sorting
        Sorting object with the selected unit splited
    """

    def __init__(self, parent_sorting, unit2split, indices_list, new_units_ids):

        self._parent_sorting = parent_sorting
        indices_list = deepcopy(indices_list)
        parents_unit_ids = parent_sorting.get_unit_ids()
        sampling_frequency = parent_sorting.get_sampling_frequency()
        unchanged_units = parents_unit_ids[parents_unit_ids!=unit2split]
        units_ids = np.concatenate([unchanged_units, new_units_ids])

        # some checks
        assert unit2split in parents_unit_ids, 'Unit to split should be in parent sorting'
        assert len(np.unique(new_units_ids))==2, 'new_units_ids must have two unique elements'
        assert not any(np.isin(new_units_ids,unchanged_units)), 'new_units_ids should be new units or one equal to unit2split'
        assert len(self._parent_sorting._sorting_segments) == len(indices_list), 'indices_list must have one element per segment'
        BaseSorting.__init__(self, sampling_frequency, units_ids)
        for si, parent_segment in enumerate(self._parent_sorting._sorting_segments):
            sub_segment = SplitSortingUnitSegment(parent_segment, unit2split, indices_list[si], new_units_ids)
            self.add_sorting_segment(sub_segment)

        # copy properties
        parent_sorting.copy_metadata(self, only_main=False, ids=None)
        
        #copy properties for unchanged units, and check if units propierties
        parent_inds = parent_sorting.ids_to_indices(unchanged_units)
        unit2split_ind = parent_sorting.id_to_index(unit2split)
        keep_units_inds = self.ids_to_indices(unchanged_units)
        split_unit_ind = self.ids_to_indices(new_units_ids)

        prop_keys = parent_sorting._properties.keys()
        for k in prop_keys:
            values = parent_sorting._properties[k]
            if values is not None:
                new_values = np.array_like(values, shape=len(units_ids))
                new_values[keep_units_inds] = values[parent_inds]
                new_values[split_unit_ind] = values[unit2split_ind]
                self.set_property(k, new_values)


        if parent_sorting.has_recording():
            self.register_recording(parent_sorting._recording)

        self._kwargs = dict(parent_sorting=parent_sorting.to_dict(), unit2split=unit2split, 
                            indices_list=indices_list, new_units_ids=new_units_ids)



class SplitSortingUnitSegment(BaseSortingSegment):
    def __init__(self, parent_segment, unit2split, indices, new_units_ids):
        BaseSortingSegment.__init__(self)
        self._parent_segment = parent_segment
        self._unit2split = unit2split
        self._new_units_ids = new_units_ids
        self._indices = indices


    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:
        if unit_id in self._new_units_ids:
            times = self._parent_segment.get_unit_spike_train(self._unit2split, start_frame=None, end_frame=None)
            if unit_id ==  self._new_units_ids[0]:
                times = times[np.logical_not(self._indices)]
            else:
                times = times[self._indices]
            if start_frame is not None:
                times = times[times >= start_frame]
            if end_frame is not None:
                times = times[times < end_frame]
        
        else:
            times = self._parent_segment.get_unit_spike_train(unit_id, start_frame, end_frame)
        
        return times
