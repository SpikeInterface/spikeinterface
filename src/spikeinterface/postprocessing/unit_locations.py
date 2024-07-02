from __future__ import annotations

import numpy as np
import warnings

from ..core.sortinganalyzer import register_result_extension, AnalyzerExtension
from .localization_tools import (
    compute_center_of_mass,
    compute_grid_convolution,
    compute_monopolar_triangulation,
)

dtype_localize_by_method = {
    "center_of_mass": [("x", "float64"), ("y", "float64")],
    "grid_convolution": [("x", "float64"), ("y", "float64"), ("z", "float64")],
    "peak_channel": [("x", "float64"), ("y", "float64")],
    "monopolar_triangulation": [("x", "float64"), ("y", "float64"), ("z", "float64"), ("alpha", "float64")],
}

possible_localization_methods = list(dtype_localize_by_method.keys())


class ComputeUnitLocations(AnalyzerExtension):
    """
    Localize units in 2D or 3D with several methods given the template.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    method: "center_of_mass" | "monopolar_triangulation" | "grid_convolution", default: "center_of_mass"
        The method to use for localization
    method_kwargs: dict, default: {}
        Other kwargs depending on the method

    Returns
    -------
    unit_locations: np.array
        unit location with shape (num_unit, 2) or (num_unit, 3) or (num_unit, 3) (with alpha)
    """

    extension_name = "unit_locations"
    depend_on = ["templates"]
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = False

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

    def _set_params(self, method="monopolar_triangulation", **method_kwargs):
        params = dict(method=method)
        params.update(method_kwargs)
        return params

    def _select_extension_data(self, unit_ids):
        unit_inds = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_unit_location = self.data["unit_locations"][unit_inds]
        return dict(unit_locations=new_unit_location)

    def _merge_extension_data(
        self, units_to_merge, new_unit_ids, new_sorting_analyzer, kept_indices=None, verbose=False, **job_kwargs
    ):
        arr = self.data["unit_locations"]
        num_dims = arr.shape[1]
        all_new_unit_ids = new_sorting_analyzer.unit_ids
        counts = self.sorting_analyzer.sorting.count_num_spikes_per_unit()
        new_unit_location = np.zeros((len(all_new_unit_ids), num_dims), dtype=arr.dtype)
        for unit_ind, unit_id in enumerate(all_new_unit_ids):
            if unit_id not in new_unit_ids:
                keep_unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                new_unit_location[unit_ind] = arr[keep_unit_index]
            else:
                id = np.flatnonzero(new_unit_ids == unit_id)[0]
                unit_ids = units_to_merge[id]
                keep_unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
                weights = np.zeros(len(unit_ids), dtype=np.float32)
                for count, id in enumerate(unit_ids):
                    weights[count] = counts[id]
                weights /= weights.sum()
                new_unit_location[unit_ind] = (arr[keep_unit_indices] * weights[:, np.newaxis]).sum(0)

        return dict(unit_locations=new_unit_location)

    def _run(self, verbose=False):
        method = self.params.get("method")
        method_kwargs = self.params.copy()
        method_kwargs.pop("method")

        assert method in possible_localization_methods

        if method == "center_of_mass":
            unit_location = compute_center_of_mass(self.sorting_analyzer, **method_kwargs)
        elif method == "grid_convolution":
            unit_location = compute_grid_convolution(self.sorting_analyzer, **method_kwargs)
        elif method == "monopolar_triangulation":
            unit_location = compute_monopolar_triangulation(self.sorting_analyzer, **method_kwargs)
        self.data["unit_locations"] = unit_location

    def get_data(self, outputs="numpy"):
        if outputs == "numpy":
            return self.data["unit_locations"]
        elif outputs == "by_unit":
            locations_by_unit = {}
            for unit_ind, unit_id in enumerate(self.sorting_analyzer.unit_ids):
                locations_by_unit[unit_id] = self.data["unit_locations"][unit_ind]
            return locations_by_unit


register_result_extension(ComputeUnitLocations)
compute_unit_locations = ComputeUnitLocations.function_factory()
