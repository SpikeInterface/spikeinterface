from typing import List, Union

import numpy as np

from .basesorting import BaseSorting, BaseSortingSegment

class MergeSorting(BaseSorting):
    """
    Class that handles merging units of a Sorting object based on a list of unit_ids.

    Parameters
    ----------
    parent_sorting: Recording
        The recording object
    units2merge: Sorting
        The sorting object
    merge_unit_id: list
        The folder where waveforms are cached
    rm_dup_delta_time: float or None
        Number of ms to consider duplicated spikes. None to don't check duplications
    cache: bool
        True to precompute duplicated events and sort times
    Returns
    -------
    sorting: Sorting
        Sorting object with the selected units merged
    """

    def __init__(self, parent_sorting, units2merge, merge_unit_id=None, rm_dup_delta_time=0.4, cache=False):
        if merge_unit_id is None:
            merge_unit_id = units2merge[0]

        self._parent_sorting = parent_sorting

        parents_unit_ids = parent_sorting.get_unit_ids()
        sampling_frequency = parent_sorting.get_sampling_frequency()

        # some checks
        assert all(u in parents_unit_ids for u in units2merge), 'units to merge are not all in parent'
        keep_units = [u for u in parents_unit_ids if u not in units2merge]
        units_ids = keep_units + [merge_unit_id]

        BaseSorting.__init__(self, sampling_frequency, units_ids)
        rm_dup_delta = int(rm_dup_delta_time / 1000 * sampling_frequency)
        for parent_segment in self._parent_sorting._sorting_segments:
            sub_segment = MergeSortingSegment(parent_segment, units2merge, merge_unit_id, rm_dup_delta, cache)
            self.add_sorting_segment(sub_segment)

        # copy properties
        parent_sorting.copy_metadata(self, only_main=False, ids=None)
        
        #copy properties for unchanged units, and check if units propierties
        keep_parent_inds = parent_sorting.ids_to_indices(keep_units)
        merge_parent_inds = parent_sorting.ids_to_indices(units2merge)
        keep_inds = self.ids_to_indices(keep_units)
        merge_ind = self.ids_to_indices([merge_unit_id])
        prop_keys = parent_sorting._properties.keys()
        for k in prop_keys:
            values = parent_sorting._properties[k]
            if values is not None:
                merge_values = values[merge_parent_inds]
                if all(merge_values==merge_values[0]): #if the merged units has the same values, just that one
                    new_values = np.array_like(values, shape=len(units_ids))
                    new_values[keep_inds] = values[keep_parent_inds]
                    new_values[merge_ind] = merge_values[0]
                    self.set_property(k, new_values)
                else:
                    self.set_property(k, None)

        if parent_sorting.has_recording():
            self.register_recording(parent_sorting._recording)

        self._kwargs = dict(parent_sorting=parent_sorting.to_dict(), units2merge=units2merge.copy(),
                            merge_unit_id=merge_unit_id, cache=cache)



class MergeSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment, units2merge, merge_unit_id, rm_dup_delta, cache):
        BaseSortingSegment.__init__(self)
        self._parent_segment = parent_segment
        self._units2merge = units2merge
        self._merge_unit_id = merge_unit_id
        self._cache = cache
        self._dup_delta = rm_dup_delta
        #if cache compute 
        if cache:
            non_dup_times  = self._parent_segment.get_unit_spike_train(units2merge[0])
            for u in units2merge[1:]:
                non_dup_times = get_non_duplicated_events(non_dup_times, 
                    self._parent_segment.get_unit_spike_train(u), rm_dup_delta)
            self._non_dup_times = non_dup_times
    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:

        if unit_id == self._merge_unit_id:
            if self._cache:
                times = self._non_dup_times[self._non_dup_times>=start_frame & self._non_dup_times<end_frame]
            else:
                #Detail here is to add extra frames in the borders to detect and remove duplicated spikes 
                if start_frame is None:
                    start_extra_frames = None
                else: 
                    start_extra_frames = start_frame-self._dup_delta
                if end_frame is None:
                    end_extra_frames = None
                else: 
                    end_extra_frames = start_frame+self._dup_delta

                non_dup_times  = self._parent_segment.get_unit_spike_train(self._units2merge[0], 
                    start_frame = start_extra_frames,end_frame=end_extra_frames)
                for u in self._units2merge[1:]:
                    non_dup_times = get_non_duplicated_events(non_dup_times, 
                        self._parent_segment.get_unit_spike_train(u, 
                            start_frame=start_extra_frames,
                            end_frame=end_extra_frames), self._dup_delta)
                
                #remove spikes on the borders
            if start_frame is not None:
                start_i = np.searchsorted(non_dup_times, start_frame, side='left')
            else:
                start_i = 0
            if end_frame is not None:
                end_i = np.searchsorted(non_dup_times, start_frame, side='right')
            else:
                end_i = len(non_dup_times)
            return non_dup_times[start_i:end_i]
        else:
            times = self._parent_segment.get_unit_spike_train(unit_id, start_frame, end_frame)

        return times

def get_non_duplicated_events(times1, times2, delta):

    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    spike_idx = np.concatenate((np.arange(times1.size, dtype='int64'), np.arange(times2.size, dtype='int64')))
    indices = times_concat.argsort(kind='mergesort')

    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    spike_index_sorted = spike_idx[indices]
    if delta is None:
        return spike_index_sorted
    inds = (np.diff(times_concat_sorted) > delta) | (np.diff(membership_sorted) == 0)

    return spike_index_sorted[np.nonzero(inds)[0]]