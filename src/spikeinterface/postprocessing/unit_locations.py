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
        params = dict(method=method, method_kwargs=method_kwargs)
        return params

    def _select_extension_data(self, unit_ids):
        unit_inds = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
        new_unit_location = self.data["unit_locations"][unit_inds]
        return dict(unit_locations=new_unit_location)

    def _run(self, verbose=False):
        method = self.params["method"]
        method_kwargs = self.params["method_kwargs"]

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
