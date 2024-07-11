from __future__ import annotations
import warnings
import numpy as np

from .core_tools import define_function_from_class
from .base import BaseExtractor
from .basesorting import BaseSorting, BaseSortingSegment


class UnitsAggregationSorting(BaseSorting):
    """
    Aggregates units of multiple sortings into a single sorting object

    Parameters
    ----------
    sorting_list: list
        List of BaseSorting objects to aggregate
    renamed_unit_ids: array-like
        If given, unit ids are renamed as provided. If None, unit ids are sequential integers.

    Returns
    -------
    aggregate_sorting: UnitsAggregationSorting
        The aggregated sorting object
    """

    def __init__(self, sorting_list, renamed_unit_ids=None):
        unit_map = {}

        num_all_units = sum([sort.get_num_units() for sort in sorting_list])
        if renamed_unit_ids is not None:
            assert len(np.unique(renamed_unit_ids)) == num_all_units, (
                "'renamed_unit_ids' doesn't have the right size" "or has duplicates!"
            )
            unit_ids = list(renamed_unit_ids)
        else:
            unit_ids_dtypes = [sort.get_unit_ids().dtype for sort in sorting_list]
            all_ids_are_same_type = np.unique(unit_ids_dtypes).size == 1
            all_units_ids_are_unique = False
            if all_ids_are_same_type:
                combined_ids = np.concatenate([sort.get_unit_ids() for sort in sorting_list])
                all_units_ids_are_unique = np.unique(combined_ids).size == num_all_units

            if all_ids_are_same_type and all_units_ids_are_unique:
                unit_ids = combined_ids
            else:
                default_unit_ids = [str(i) for i in range(num_all_units)]
                if all_ids_are_same_type and np.issubdtype(unit_ids_dtypes[0], np.integer):
                    unit_ids = np.arange(num_all_units, dtype=np.uint64)
                else:
                    unit_ids = default_unit_ids

        # unit map maps unit ids that are used to get spike trains
        u_id = 0
        for s_i, sorting in enumerate(sorting_list):
            single_unit_ids = sorting.get_unit_ids()
            for unit_id in single_unit_ids:
                unit_map[unit_ids[u_id]] = {"sorting_id": s_i, "unit_id": unit_id}
                u_id += 1

        sampling_frequency = sorting_list[0].get_sampling_frequency()
        num_segments = sorting_list[0].get_num_segments()

        ok1 = all(sampling_frequency == sort.get_sampling_frequency() for sort in sorting_list)
        ok2 = all(num_segments == sort.get_num_segments() for sort in sorting_list)
        if not (ok1 and ok2):
            raise ValueError("Sortings don't have the same sampling_frequency/num_segments")

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        annotation_keys = sorting_list[0].get_annotation_keys()
        for annotation_name in annotation_keys:
            if not all([annotation_name in sort.get_annotation_keys() for sort in sorting_list]):
                continue

            annotations = np.array([sort.get_annotation(annotation_name, copy=False) for sort in sorting_list])
            if np.all(annotations == annotations[0]):
                self.set_annotation(annotation_name, sorting_list[0].get_annotation(annotation_name))

        property_keys = {}
        property_dict = {}
        deleted_keys = []
        for sort in sorting_list:
            for prop_name in sort.get_property_keys():
                if prop_name in deleted_keys:
                    continue
                if prop_name in property_keys:
                    if property_keys[prop_name] != sort.get_property(prop_name).dtype:
                        print(f"Skipping property '{prop_name}: difference in dtype between sortings'")
                        del property_keys[prop_name]
                        deleted_keys.append(prop_name)
                else:
                    property_keys[prop_name] = sort.get_property(prop_name).dtype
        for prop_name in property_keys:
            dtype = property_keys[prop_name]
            property_dict[prop_name] = np.array([], dtype=dtype)

            for sort in sorting_list:
                if prop_name in sort.get_property_keys():
                    values = sort.get_property(prop_name)
                else:
                    if dtype.kind not in BaseExtractor.default_missing_property_values:
                        del property_dict[prop_name]
                        break
                    values = np.full(
                        sort.get_num_units(), BaseExtractor.default_missing_property_values[dtype.kind], dtype=dtype
                    )

                try:
                    property_dict[prop_name] = np.concatenate((property_dict[prop_name], values))
                except Exception as e:
                    print(f"Skipping property '{prop_name}' due to shape inconsistency")
                    del property_dict[prop_name]
                    break
        for prop_name, prop_values in property_dict.items():
            self.set_property(key=prop_name, values=prop_values)

        # add segments
        for i_seg in range(num_segments):
            parent_segments = [sort._sorting_segments[i_seg] for sort in sorting_list]
            sub_segment = UnitsAggregationSortingSegment(unit_map, parent_segments)
            self.add_sorting_segment(sub_segment)

        self._sortings = sorting_list
        self._kwargs = {"sorting_list": sorting_list, "renamed_unit_ids": renamed_unit_ids}

    @property
    def sortings(self):
        return self._sortings


class UnitsAggregationSortingSegment(BaseSortingSegment):
    def __init__(self, unit_map, parent_segments):
        BaseSortingSegment.__init__(self)
        self._unit_map = unit_map
        self._parent_segments = parent_segments

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> np.ndarray:
        sorting_id = self._unit_map[unit_id]["sorting_id"]
        unit_id_sorting = self._unit_map[unit_id]["unit_id"]
        times = self._parent_segments[sorting_id].get_unit_spike_train(unit_id_sorting, start_frame, end_frame)
        return times


aggregate_units = define_function_from_class(UnitsAggregationSorting, "aggregate_units")
