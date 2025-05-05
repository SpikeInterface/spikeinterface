from __future__ import annotations
from typing import Union
import numpy as np
from spikeinterface.core.basesorting import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class
from copy import deepcopy
from spikeinterface.core.sorting_tools import generate_unit_ids_for_merge_group


class MergeUnitsSorting(BaseSorting):
    """
    Class that handles several merges of units from a Sorting object based on a list of lists of unit_ids.

    Parameters
    ----------
    sorting : BaseSorting
        The sorting object
    units_to_merge : list/tuple of lists/tuples
        A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
        but it can also have more (merge multiple units at once).
    new_unit_ids : None or list
        A new unit_ids for merged units. If given, it needs to have the same length as `units_to_merge`
    properties_policy : "keep" | "remove", default: "keep"
        Policy used to propagate properties. If "keep" the properties will be passed to the new units
         (if the units_to_merge have the same value). If "remove" the new units will have an empty
         value for all the properties of the new unit.
    delta_time_ms : float or None
        Number of ms to consider for duplicated spikes. None won't check for duplications

    Returns
    -------
    sorting : Sorting
        Sorting object with the selected units merged
    """

    def __init__(self, sorting, units_to_merge, new_unit_ids=None, properties_policy="keep", delta_time_ms=0.4):
        self._parent_sorting = sorting

        if not isinstance(units_to_merge[0], (list, tuple)):
            # keep backward compatibility : the previous behavior was only one merge
            units_to_merge = [units_to_merge]

        num_merge = len(units_to_merge)

        parents_unit_ids = sorting.unit_ids
        sampling_frequency = sorting.get_sampling_frequency()

        new_unit_ids = generate_unit_ids_for_merge_group(
            sorting.unit_ids, units_to_merge, new_unit_ids=new_unit_ids, new_id_strategy="append"
        )

        all_removed_ids = []
        for ids in units_to_merge:
            all_removed_ids.extend(ids)
        keep_unit_ids = [u for u in parents_unit_ids if u not in all_removed_ids]

        assert len(new_unit_ids) == num_merge, "new_unit_ids must have the same size as units_to_merge"

        # some checks
        for ids in units_to_merge:
            assert all(u in parents_unit_ids for u in ids), "units to merge are not all in parent"
        assert properties_policy in ("keep", "remove"), "properties_policy must be " "keep" " or " "remove" ""

        # new units are put at the end
        unit_ids = keep_unit_ids + list(new_unit_ids)
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        # assert all(np.isin(keep_unit_ids, self.unit_ids)), 'new_unit_id should have a compatible format with the parent ids'

        if delta_time_ms is None:
            rm_dup_delta = None
        else:
            rm_dup_delta = int(delta_time_ms / 1000 * sampling_frequency)
        for parent_segment in self._parent_sorting._sorting_segments:
            sub_segment = MergeUnitsSortingSegment(parent_segment, units_to_merge, new_unit_ids, rm_dup_delta)
            self.add_sorting_segment(sub_segment)

        ann_keys = sorting._annotations.keys()
        self._annotations = deepcopy({k: sorting._annotations[k] for k in ann_keys})

        # copy properties for unchanged units, and check if units propierties are the same
        keep_parent_inds = sorting.ids_to_indices(keep_unit_ids)
        # ~ all_removed_inds = parent_sorting.ids_to_indices(all_removed_ids)
        keep_inds = self.ids_to_indices(keep_unit_ids)
        # ~ merge_inds = self.ids_to_indices(new_unit_ids)
        prop_keys = sorting.get_property_keys()
        for key in prop_keys:
            parent_values = sorting.get_property(key)

            if properties_policy == "keep":
                # propagate keep values
                shape = (len(unit_ids),) + parent_values.shape[1:]
                new_values = np.empty(shape=shape, dtype=parent_values.dtype)
                new_values[keep_inds] = parent_values[keep_parent_inds]
                for new_id, ids in zip(new_unit_ids, units_to_merge):
                    removed_inds = sorting.ids_to_indices(ids)
                    merge_values = parent_values[removed_inds]

                    same_property_values = np.all([np.array_equal(m, merge_values[0]) for m in merge_values[1:]])

                    ind = self.id_to_index(new_id)
                    if same_property_values:
                        # and new values only if they are all similar
                        new_values[ind] = merge_values[0]
                    else:
                        if parent_values.dtype.kind == "f":
                            new_values[ind] = np.nan
                        elif parent_values.dtype.kind in ("U", "S"):
                            new_values[ind] = ""
                        else:
                            new_values = new_values.astype(object)
                            new_values[ind] = None
                self.set_property(key, new_values)

            elif properties_policy == "remove":
                self.set_property(key, parent_values[keep_parent_inds], keep_unit_ids)

        if sorting.has_recording():
            self.register_recording(sorting._recording)

        # make it jsonable
        units_to_merge = [list(e) for e in units_to_merge]
        self._kwargs = dict(
            sorting=sorting,
            units_to_merge=units_to_merge,
            new_unit_ids=new_unit_ids,
            properties_policy=properties_policy,
            delta_time_ms=delta_time_ms,
        )


merge_units_sorting = define_function_from_class(source_class=MergeUnitsSorting, name="merge_units_sorting")


class MergeUnitsSortingSegment(BaseSortingSegment):
    def __init__(self, parent_segment, units_to_merge, new_unit_ids, rm_dup_delta):
        BaseSortingSegment.__init__(self)
        self._parent_segment = parent_segment
        self._units_to_merge = units_to_merge
        self.new_unit_ids = new_unit_ids
        self._dup_delta = rm_dup_delta
        # if cache compute
        self._merged_spike_times = []
        for ids in units_to_merge:
            spike_times = get_non_duplicated_events(
                [self._parent_segment.get_unit_spike_train(u, None, None) for u in ids], rm_dup_delta
            )
            self._merged_spike_times.append(spike_times)

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
    ) -> np.ndarray:
        if unit_id in self.new_unit_ids:
            ind = self.new_unit_ids.index(unit_id)
            spike_times = self._merged_spike_times[ind]

            if start_frame is not None:
                start_i = np.searchsorted(spike_times, start_frame, side="left")
            else:
                start_i = 0
            if end_frame is not None:
                end_i = np.searchsorted(spike_times, start_frame, side="right")
            else:
                end_i = len(spike_times)
            return spike_times[start_i:end_i]
        else:
            spike_times = self._parent_segment.get_unit_spike_train(unit_id, start_frame, end_frame)
            return spike_times


# TODO move this function to postprocessing or similar
def get_non_duplicated_events(times_list, delta):
    times_concat = np.concatenate(times_list)
    if len(times_concat) == 0:
        return times_concat
    indices = times_concat.argsort(kind="mergesort")
    times_concat_sorted = times_concat[indices]

    if delta is None:
        return times_concat_sorted
    membership = np.concatenate([np.ones(t.shape) * i for i, t in enumerate(times_list)])
    membership_sorted = membership[indices]

    inds = np.nonzero((np.diff(times_concat_sorted) > delta) | (np.diff(membership_sorted) == 0))[0]
    # always add the first one and realing counting
    inds = np.concatenate([[0], inds + 1])
    return times_concat_sorted[inds]
