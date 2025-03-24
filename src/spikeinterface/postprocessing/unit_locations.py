from __future__ import annotations

import numpy as np
import warnings

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from .localization_tools import _unit_location_methods


# this dict is for peak location
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
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object
    method : "monopolar_triangulation" |  "center_of_mass" | "grid_convolution", default: "monopolar_triangulation"
        The method to use for localization
    **method_kwargs : dict, default: {}
        Kwargs which are passed to the method function. These can be found in the docstrings of `compute_center_of_mass`, `compute_grid_convolution` and `compute_monopolar_triangulation`.

    Returns
    -------
    unit_locations : np.array
        unit location with shape (num_unit, 2) or (num_unit, 3) or (num_unit, 3) (with alpha)
    """

    extension_name = "unit_locations"
    depend_on = ["templates"]
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False
    need_backward_compatibility_on_load = True

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

    def _handle_backward_compatibility_on_load(self):
        if "method_kwargs" in self.params:
            # make compatible analyzer created between february 24 and july 24
            method_kwargs = self.params.pop("method_kwargs")
            self.params.update(**method_kwargs)

    def _set_params(self, method="monopolar_triangulation", **method_kwargs):
        params = dict(method=method)
        params.update(method_kwargs)
        return params

    def _select_extension_data(self, unit_ids):
        unit_inds = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_unit_location = self.data["unit_locations"][unit_inds]
        return dict(unit_locations=new_unit_location)

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        old_unit_locations = self.data["unit_locations"]
        num_dims = old_unit_locations.shape[1]

        method = self.params.get("method")
        method_kwargs = self.params.copy()
        method_kwargs.pop("method")
        func = _unit_location_methods[method]
        new_unit_locations = func(new_sorting_analyzer, unit_ids=new_unit_ids, **method_kwargs)
        assert new_unit_locations.shape[0] == len(new_unit_ids)

        all_new_unit_ids = new_sorting_analyzer.unit_ids
        unit_location = np.zeros((len(all_new_unit_ids), num_dims), dtype=old_unit_locations.dtype)
        for unit_index, unit_id in enumerate(all_new_unit_ids):
            if unit_id not in new_unit_ids:
                old_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                unit_location[unit_index] = old_unit_locations[old_index]
            else:
                new_index = list(new_unit_ids).index(unit_id)
                unit_location[unit_index] = new_unit_locations[new_index]

        return dict(unit_locations=unit_location)

    def _run(self, verbose=False):
        method = self.params.get("method")
        method_kwargs = self.params.copy()
        method_kwargs.pop("method")

        if method not in _unit_location_methods:
            raise ValueError(f"Wrong method for unit_locations : it should be in {list(_unit_location_methods.keys())}")

        func = _unit_location_methods[method]
        self.data["unit_locations"] = func(self.sorting_analyzer, **method_kwargs)

    def get_data(self, outputs="numpy"):
        if outputs == "numpy":
            return self.data["unit_locations"]
        elif outputs == "by_unit":
            locations_by_unit = {}
            for unit_index, unit_id in enumerate(self.sorting_analyzer.unit_ids):
                locations_by_unit[unit_id] = self.data["unit_locations"][unit_index]
            return locations_by_unit


register_result_extension(ComputeUnitLocations)
compute_unit_locations = ComputeUnitLocations.function_factory()
