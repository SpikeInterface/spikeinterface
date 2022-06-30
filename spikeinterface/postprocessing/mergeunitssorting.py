from typing import List, Union
import numpy as np
from spikeinterface.core.basesorting import BaseSorting, BaseSortingSegment
from copy import deepcopy

class MergeUnitsSorting(BaseSorting):
    """
    Class that handles merging units of a Sorting object based on a list of unit_ids.

    Parameters
    ----------
    parent_sorting: Recording
        The recording object
    units_to_merge: Sorting
        The sorting object
    new_unit_id: list
        Unit id to use for the combined unit. It could be one for the merged ones.
    properties_policy: str
        Policy used to propagate propierties. If 'keep' the properties will be pass to the new units
         (if the units_to_merge have the same value). If 'remove' the new units will have an empty 
         value for all the properties of the new unit.
         Default: 'keep'
    delta_time_ms: float or None
        Number of ms to consider duplicated spikes. None to don't check duplications
    Returns
    -------
    sorting: Sorting
        Sorting object with the selected units merged
    """

    def __init__(self, parent_sorting, units_to_merge, new_unit_id=None, properties_policy='keep', delta_time_ms=0.4):

        self._parent_sorting = parent_sorting

        parents_unit_ids = parent_sorting.get_unit_ids()
        sampling_frequency = parent_sorting.get_sampling_frequency()

        if new_unit_id is None:
            #select new_units_ids grater that the max id, event greater than the numerical str ids 
            if np.issubdtype(parents_unit_ids.dtype, np.character):
                new_unit_id = max([0]+[int(p) for p in parents_unit_ids if p.isdigit()])+1
            else:
                new_unit_id = max(parents_unit_ids)+1
        else:
            assert new_unit_id not in parents_unit_ids, 'new_unit_id already in parent_sorting'
        # some checks
        assert all(u in parents_unit_ids for u in units_to_merge), 'units to merge are not all in parent'
        assert properties_policy=='keep' or properties_policy=='remove', 'properties_policy must be ''keep'' or ''remove'''
        keep_units = [u for u in parents_unit_ids if u not in units_to_merge]
        units_ids = keep_units + [new_unit_id] 
        BaseSorting.__init__(self, sampling_frequency, units_ids)
        assert all(np.isin(keep_units, self.unit_ids)), 'new_unit_id should have a compatible format with the parent ids'
        if delta_time_ms is None:
            rm_dup_delta = None
        else:
            rm_dup_delta = int(delta_time_ms / 1000 * sampling_frequency)
        for parent_segment in self._parent_sorting._sorting_segments:
            sub_segment = MergeUnitsSortingSegment(parent_segment, units_to_merge, new_unit_id, rm_dup_delta)
            self.add_sorting_segment(sub_segment)

        ann_keys = parent_sorting._annotations.keys()
        self._annotations = deepcopy({k: parent_sorting._annotations[k] for k in ann_keys})
                
        #copy properties for unchanged units, and check if units propierties are the same
        keep_parent_inds = parent_sorting.ids_to_indices(keep_units)
        merge_parent_inds = parent_sorting.ids_to_indices(units_to_merge)
        keep_inds = self.ids_to_indices(keep_units)
        merge_ind = self.ids_to_indices([new_unit_id])
        prop_keys = parent_sorting._properties.keys()
        for k in prop_keys:
            values = parent_sorting._properties[k]
            if properties_policy=='keep':
                merge_values = values[merge_parent_inds]
                if all(merge_values==merge_values[0]):
                    new_values = np.empty_like(values, shape=len(units_ids))
                    new_values[merge_ind] = merge_values[0]
                    if len(keep_inds)>0:
                        new_values[keep_inds] = values[keep_parent_inds]
                    self.set_property(k, new_values)
                    continue
            self.set_property(k, values[keep_parent_inds], keep_units)

        if parent_sorting.has_recording():
            self.register_recording(parent_sorting._recording)

        self._kwargs = dict(parent_sorting=parent_sorting.to_dict(), units_to_merge=list(units_to_merge),
                            new_unit_id=new_unit_id, properties_policy=properties_policy, delta_time_ms=delta_time_ms)



class MergeUnitsSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment, units_to_merge, new_unit_id, rm_dup_delta):
        BaseSortingSegment.__init__(self)
        self._parent_segment = parent_segment
        self._units_to_merge = units_to_merge
        self._new_unit_id = new_unit_id
        self._dup_delta = rm_dup_delta
        #if cache compute 
        self._merged_spike_times = get_non_duplicated_events([self._parent_segment.get_unit_spike_train(u,None,None) for u in units_to_merge], rm_dup_delta)

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:

        if unit_id == self._new_unit_id:

            if start_frame is not None:
                start_i = np.searchsorted(self._merged_spike_times, start_frame, side='left')
            else:
                start_i = 0
            if end_frame is not None:
                end_i = np.searchsorted(self._merged_spike_times, start_frame, side='right')
            else:
                end_i = len(self._merged_spike_times)
            return self._merged_spike_times[start_i:end_i]
        else:
            times = self._parent_segment.get_unit_spike_train(unit_id, start_frame, end_frame)
            return times

#TODO move this function to postprocessing or similar
def get_non_duplicated_events(times_list, delta): 

    times_concat = np.concatenate(times_list)
    if len(times_concat)==0:
        return times_concat
    indices = times_concat.argsort(kind='mergesort')
    times_concat_sorted = times_concat[indices]
    
    if delta is None:
        return times_concat_sorted
    membership = np.concatenate([np.ones(t.shape)*i for i,t in enumerate(times_list)])    
    membership_sorted = membership[indices]   
    
    inds = np.nonzero((np.diff(times_concat_sorted) > delta) | (np.diff(membership_sorted) == 0))[0]
    #always add the first one and realing counting
    inds = np.concatenate([[0],inds+1])
    return times_concat_sorted[inds]
