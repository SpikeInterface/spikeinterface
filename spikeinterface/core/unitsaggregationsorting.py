from typing import List, Union

import numpy as np

from .basesorting import BaseSorting, BaseSortingSegment


class UnitsAggregationSorting(BaseSorting):
    """
    Class that handles aggregating units from different sortings, e.g. from different channel groups.

    Do not use this class directly but use `si.aggregate_units(...)`

    """
    def __init__(self, sorting_list):
        unit_map = {}

        # units are renamed from 0 to N-1
        unit_ids = []
        u_id = 0
        for s_i, sorting in enumerate(sorting_list):
            single_unit_ids = sorting.get_unit_ids()
            for unit_id in single_unit_ids:
                unit_ids.append(u_id)
                unit_map[u_id] = {'sorting_id': s_i, 'unit_id': unit_id}
                u_id += 1

        sampling_frequency = sorting_list[0].get_sampling_frequency()
        num_segments = sorting_list[0].get_num_segments()

        ok1 = all(sampling_frequency == sort.get_sampling_frequency() for sort in sorting_list)
        ok2 = all(num_segments == sort.get_num_segments() for sort in sorting_list)
        if not (ok1 and ok2):
            raise ValueError("Sortings don't have the same sampling_frequency/num_segments")

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        for i_seg in range(num_segments):
            parent_segments = [sort._sorting_segments[i_seg] for sort in sorting_list]
            sub_segment = UnitsAggregationSortingSegment(unit_map, parent_segments)
            self.add_sorting_segment(sub_segment)

        property_keys = sorting_list[0].get_property_keys()
        property_dict = {}
        for prop_name in property_keys:
            if all([prop_name in sort.get_property_keys() for sort in sorting_list]):
                property_dict[prop_name] = np.array([])
                for sort in sorting_list:
                    property_dict[prop_name] = np.concatenate((property_dict[prop_name], sort.get_property(prop_name)))

        for prop_name, prop_values in property_dict.items():
            self.set_property(key=prop_name, values=prop_values)

        self._sortings = sorting_list
        self._kwargs = {'sorting_list': [sort.to_dict() for sort in sorting_list]}

    @property
    def sortings(self):
        return self._sortings


class UnitsAggregationSortingSegment(BaseSortingSegment):
    def __init__(self, unit_map, parent_segments):
        BaseSortingSegment.__init__(self)
        self._unit_map = unit_map
        self._parent_segments = parent_segments

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame: Union[int, None] = None,
                             end_frame: Union[int, None] = None,
                             ) -> np.ndarray:
        sorting_id = self._unit_map[unit_id]['sorting_id']
        unit_id_sorting = self._unit_map[unit_id]['unit_id']
        times = self._parent_segments[sorting_id].get_unit_spike_train(unit_id_sorting, start_frame, end_frame)
        return times


def aggregate_units(sorting_list):
    """
    Aggregates units of multiple sortings into a single sorting object

    Parameters
    ----------
    sorting_list: list
        List of BaseSorting objects to aggregate

    Returns
    -------
    aggregate_sortimg: UnitsAggregationSorting
        The aggregated sorting object
    """
    return UnitsAggregationSorting(sorting_list)
