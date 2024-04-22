from __future__ import annotations
from typing import List, Union

import numpy as np
from copy import deepcopy
from spikeinterface.core.basesorting import BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class


class SplitUnitSorting(BaseSorting):
    """
    Class that handles spliting of a unit. It creates a new Sorting object linked to parent_sorting.

    Parameters
    ----------
    parent_sorting: Recording
        The recording object
    parent_unit_id: int
        Unit id of the unit to split
    indices_list: list
        A list of index arrays selecting the spikes to split in each segment.
        Each array can contain more than 2 indices (e.g. for splitting in 3 or more units) and it should
        be the same length as the spike train (for each segment)
    new_unit_ids: int
        Unit ids of the new units to be created.
    properties_policy: "keep" | "remove", default: "keep"
        Policy used to propagate properties. If "keep" the properties will be passed to the new units
         (if the units_to_merge have the same value). If "remove" the new units will have an empty
         value for all the properties of the new unit.
    Returns
    -------
    sorting: Sorting
        Sorting object with the selected units split
    """

    def __init__(self, parent_sorting, split_unit_id, indices_list, new_unit_ids=None, properties_policy="keep"):
        if type(indices_list) is not list:
            indices_list = [indices_list]
        parents_unit_ids = parent_sorting.get_unit_ids()
        tot_splits = max([v.max() for v in indices_list]) + 1
        unchanged_units = parents_unit_ids[parents_unit_ids != split_unit_id]

        if new_unit_ids is None:
            # select new_unit_ids greater that the max id, event greater than the numerical str ids
            if np.issubdtype(parents_unit_ids.dtype, np.character):
                new_unit_ids = max([0] + [int(p) for p in parents_unit_ids if p.isdigit()]) + 1
            else:
                new_unit_ids = max(parents_unit_ids) + 1
            new_unit_ids = np.array([u + new_unit_ids for u in range(tot_splits)], dtype=parents_unit_ids.dtype)
        else:
            new_unit_ids = np.array(new_unit_ids, dtype=parents_unit_ids.dtype)
            assert len(np.unique(new_unit_ids)) == len(new_unit_ids), "Each element in new_unit_ids must be unique"
            assert len(new_unit_ids) <= tot_splits, "indices_list has more id indices than the length of new_unit_ids"

        assert parent_sorting.get_num_segments() == len(
            indices_list
        ), "The length of indices_list must be the same as parent_sorting.get_num_segments"
        assert split_unit_id in parents_unit_ids, "Unit to split must be in parent sorting"
        assert properties_policy == "keep" or properties_policy == "remove", (
            "properties_policy must be " "keep" " or " "remove" ""
        )
        assert not any(
            np.isin(new_unit_ids, unchanged_units)
        ), "new_unit_ids should be new unit ids or no more than one unit id can be found in split_unit_id"

        sampling_frequency = parent_sorting.get_sampling_frequency()
        units_ids = np.concatenate([unchanged_units, new_unit_ids])

        self._parent_sorting = parent_sorting
        indices_list = deepcopy(indices_list)

        BaseSorting.__init__(self, sampling_frequency, units_ids)
        assert all(
            np.isin(unchanged_units, self.unit_ids)
        ), "new_unit_ids should have a compatible format with the parent ids"

        for si, parent_segment in enumerate(self._parent_sorting._sorting_segments):
            sub_segment = SplitSortingUnitSegment(parent_segment, split_unit_id, indices_list[si], new_unit_ids)
            self.add_sorting_segment(sub_segment)

        # copy properties
        ann_keys = parent_sorting._annotations.keys()
        self._annotations = deepcopy({k: parent_sorting._annotations[k] for k in ann_keys})

        # copy properties for unchanged units, and check if units propierties
        keep_parent_inds = parent_sorting.ids_to_indices(unchanged_units)
        split_unit_id_ind = parent_sorting.id_to_index(split_unit_id)
        keep_units_inds = self.ids_to_indices(unchanged_units)
        split_unit_ind = self.ids_to_indices(new_unit_ids)
        # copy properties from original units to split ones
        prop_keys = parent_sorting._properties.keys()
        for k in prop_keys:
            values = parent_sorting._properties[k]
            if properties_policy == "keep":
                new_values = np.empty_like(values, shape=len(units_ids))
                new_values[keep_units_inds] = values[keep_parent_inds]
                new_values[split_unit_ind] = values[split_unit_id_ind]
                self.set_property(k, new_values)
                continue
            self.set_property(k, values[keep_parent_inds], unchanged_units)

        if parent_sorting.has_recording():
            self.register_recording(parent_sorting._recording)

        self._kwargs = dict(
            parent_sorting=parent_sorting,
            split_unit_id=split_unit_id,
            indices_list=indices_list,
            new_unit_ids=new_unit_ids,
            properties_policy=properties_policy,
        )


split_unit_sorting = define_function_from_class(source_class=SplitUnitSorting, name="split_unit_sorting")


class SplitSortingUnitSegment(BaseSortingSegment):
    def __init__(self, parent_segment, split_unit_id, indices, new_unit_ids):
        BaseSortingSegment.__init__(self)
        self._parent_segment = parent_segment
        self._new_unit_ids = new_unit_ids
        self._spike_trains = dict()
        times = self._parent_segment.get_unit_spike_train(split_unit_id, start_frame=None, end_frame=None)
        for idx, unit_id in enumerate(self._new_unit_ids):
            self._spike_trains[unit_id] = times[indices == idx]

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
    ) -> np.ndarray:
        if unit_id in self._new_unit_ids:
            if start_frame is None:
                init = 0
            else:
                init = np.searchsorted(self._spike_trains[unit_id], start_frame, side="left")
            if end_frame is None:
                endf = len(self._spike_trains[unit_id])
            else:
                endf = np.searchsorted(self._spike_trains[unit_id], end_frame, side="right")
            times = self._spike_trains[unit_id][init:endf]
        else:
            times = self._parent_segment.get_unit_spike_train(unit_id, start_frame, end_frame)

        return times
