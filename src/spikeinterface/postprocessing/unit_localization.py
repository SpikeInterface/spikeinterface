from __future__ import annotations

import numpy as np
import warnings

from ..core.sortinganalyzer import register_result_extension, AnalyzerExtension
from ..core import compute_sparsity
from ..core.template_tools import get_template_extremum_channel, _get_nbefore, _get_dense_templates_array
from .localization_tools import (
    make_radial_order_parents,
    solve_monopolar_triangulation,
    get_grid_convolution_templates_and_weights,
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
    depend_on = [
        "fast_templates|templates",
    ]
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

    def _run(self):
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


def compute_monopolar_triangulation(
    sorting_analyzer,
    optimizer="minimize_with_log_penality",
    radius_um=75,
    max_distance_um=1000,
    return_alpha=False,
    enforce_decrease=False,
    feature="ptp",
):
    """
    Localize unit with monopolar triangulation.
    This method is from Julien Boussard, Erdem Varol and Charlie Windolf
    https://www.biorxiv.org/content/10.1101/2021.11.05.467503v1

    There are 2 implementations of the 2 optimizer variants:
      * https://github.com/int-brain-lab/spikes_localization_registration/blob/main/localization_pipeline/localizer.py
      * https://github.com/cwindolf/spike-psvae/blob/main/spike_psvae/localization.py

    Important note about axis:
      * x/y are dimmension on the probe plane (dim0, dim1)
      * y is the depth by convention
      * z it the orthogonal axis to the probe plan (dim2)

    Code from Erdem, Julien and Charlie do not use the same convention!!!


    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    method: "least_square" | "minimize_with_log_penality", default: "least_square"
       The optimizer to use
    radius_um: float, default: 75
        For channel sparsity
    max_distance_um: float, default: 1000
        to make bounddary in x, y, z and also for alpha
    return_alpha: bool, default: False
        Return or not the alpha value
    enforce_decrease : bool, default: False
        Enforce spatial decreasingness for PTP vectors
    feature: "ptp" | "energy" | "peak_voltage", default: "ptp"
        The available features to consider for estimating the position via
        monopolar triangulation are peak-to-peak amplitudes ("ptp", default),
        energy ("energy", as L2 norm) or voltages at the center of the waveform
        ("peak_voltage")

    Returns
    -------
    unit_location: np.array
        3d or 4d, x, y, z, alpha
        alpha is the amplitude at source estimation
    """
    assert optimizer in ("least_square", "minimize_with_log_penality")

    assert feature in ["ptp", "energy", "peak_voltage"], f"{feature} is not a valid feature"
    unit_ids = sorting_analyzer.unit_ids

    contact_locations = sorting_analyzer.get_channel_locations()

    sparsity = compute_sparsity(sorting_analyzer, method="radius", radius_um=radius_um)
    templates = _get_dense_templates_array(sorting_analyzer)
    nbefore = _get_nbefore(sorting_analyzer)

    if enforce_decrease:
        neighbours_mask = np.zeros((templates.shape[0], templates.shape[2]), dtype=bool)
        for i, unit_id in enumerate(unit_ids):
            chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
            neighbours_mask[i, chan_inds] = True
        enforce_decrease_radial_parents = make_radial_order_parents(contact_locations, neighbours_mask)
        best_channels = get_template_extremum_channel(sorting_analyzer, outputs="index")

    unit_location = np.zeros((unit_ids.size, 4), dtype="float64")
    for i, unit_id in enumerate(unit_ids):
        chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
        local_contact_locations = contact_locations[chan_inds, :]

        # wf is (nsample, nchan) - chann is only nieghboor
        wf = templates[i, :, :][:, chan_inds]
        if feature == "ptp":
            wf_data = wf.ptp(axis=0)
        elif feature == "energy":
            wf_data = np.linalg.norm(wf, axis=0)
        elif feature == "peak_voltage":
            wf_data = np.abs(wf[nbefore])

        # if enforce_decrease:
        #    enforce_decrease_shells_data(
        #        wf_data, best_channels[unit_id], enforce_decrease_radial_parents, in_place=True
        #    )

        unit_location[i] = solve_monopolar_triangulation(wf_data, local_contact_locations, max_distance_um, optimizer)

    if not return_alpha:
        unit_location = unit_location[:, :3]

    return unit_location


def compute_center_of_mass(sorting_analyzer, peak_sign="neg", radius_um=75, feature="ptp"):
    """
    Computes the center of mass (COM) of a unit based on the template amplitudes.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute best channels
    radius_um: float
        Radius to consider in order to estimate the COM
    feature: "ptp" | "mean" | "energy" | "peak_voltage", default: "ptp"
        Feature to consider for computation

    Returns
    -------
    unit_location: np.array
    """
    unit_ids = sorting_analyzer.unit_ids

    contact_locations = sorting_analyzer.get_channel_locations()

    assert feature in ["ptp", "mean", "energy", "peak_voltage"], f"{feature} is not a valid feature"

    sparsity = compute_sparsity(sorting_analyzer, peak_sign=peak_sign, method="radius", radius_um=radius_um)
    templates = _get_dense_templates_array(sorting_analyzer)
    nbefore = _get_nbefore(sorting_analyzer)

    unit_location = np.zeros((unit_ids.size, 2), dtype="float64")
    for i, unit_id in enumerate(unit_ids):
        chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
        local_contact_locations = contact_locations[chan_inds, :]

        wf = templates[i, :, :]

        if feature == "ptp":
            wf_data = (wf[:, chan_inds]).ptp(axis=0)
        elif feature == "mean":
            wf_data = (wf[:, chan_inds]).mean(axis=0)
        elif feature == "energy":
            wf_data = np.linalg.norm(wf[:, chan_inds], axis=0)
        elif feature == "peak_voltage":
            wf_data = wf[nbefore, chan_inds]

        # center of mass
        com = np.sum(wf_data[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_data)
        unit_location[i, :] = com

    return unit_location


def compute_grid_convolution(
    sorting_analyzer,
    peak_sign="neg",
    radius_um=40.0,
    upsampling_um=5,
    sigma_ms=0.25,
    margin_um=50,
    prototype=None,
    percentile=5,
    weight_method={},
):
    """
    Estimate the positions of the templates from a large grid of fake templates

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute best channels
    radius_um: float, default: 40.0
        Radius to consider for the fake templates
    upsampling_um: float, default: 5
        Upsampling resolution for the grid of templates
    sigma_ms: float, default: 0.25
        The temporal decay of the fake templates
    margin_um: float, default: 50
        The margin for the grid of fake templates
    prototype: np.array or None, default: None
        Fake waveforms for the templates. If None, generated as Gaussian
    percentile: float, default: 5
        The percentage  in [0, 100] of the best scalar products kept to
        estimate the position
    weight_method: dict
        Parameter that should be provided to the get_convolution_weights() function
        in order to know how to estimate the positions. One argument is mode that could
        be either gaussian_2d (KS like) or exponential_3d (default)
    Returns
    -------
    unit_location: np.array
    """

    contact_locations = sorting_analyzer.get_channel_locations()
    unit_ids = sorting_analyzer.unit_ids

    templates = _get_dense_templates_array(sorting_analyzer)
    nbefore = _get_nbefore(sorting_analyzer)
    nafter = templates.shape[1] - nbefore

    fs = sorting_analyzer.sampling_frequency
    percentile = 100 - percentile
    assert 0 <= percentile <= 100, "Percentile should be in [0, 100]"

    time_axis = np.arange(-nbefore, nafter) * 1000 / fs
    if prototype is None:
        prototype = np.exp(-(time_axis**2) / (2 * (sigma_ms**2)))
        if peak_sign == "neg":
            prototype *= -1

    prototype = prototype[:, np.newaxis]

    template_positions, weights, nearest_template_mask, z_factors = get_grid_convolution_templates_and_weights(
        contact_locations, radius_um, upsampling_um, margin_um, weight_method
    )

    peak_channels = get_template_extremum_channel(sorting_analyzer, peak_sign, outputs="index")

    weights_sparsity_mask = weights > 0

    nb_weights = weights.shape[0]
    unit_location = np.zeros((unit_ids.size, 3), dtype="float64")

    for i, unit_id in enumerate(unit_ids):
        main_chan = peak_channels[unit_id]
        wf = templates[i, :, :]
        nearest_mask = nearest_template_mask[main_chan, :]
        channel_mask = np.sum(weights_sparsity_mask[:, :, nearest_mask], axis=(0, 2)) > 0
        num_templates = np.sum(nearest_mask)
        sub_w = weights[:, channel_mask, :][:, :, nearest_mask]
        global_products = (wf[:, channel_mask] * prototype).sum(axis=0)

        dot_products = np.zeros((nb_weights, num_templates), dtype=np.float32)
        for count in range(nb_weights):
            dot_products[count] = np.dot(global_products, sub_w[count])

        mask = dot_products < 0
        if percentile > 0:
            dot_products[mask] = np.nan
            ## We need to catch warnings because some line can have only NaN, and
            ## if so the nanpercentile function throws a warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                thresholds = np.nanpercentile(dot_products, percentile)
            thresholds = np.nan_to_num(thresholds)
            dot_products[dot_products < thresholds] = 0
        dot_products[mask] = 0

        nearest_templates = template_positions[nearest_mask]
        for count in range(nb_weights):
            unit_location[i, :2] += np.dot(dot_products[count], nearest_templates)

        scalar_products = dot_products.sum(1)
        unit_location[i, 2] = np.dot(z_factors, scalar_products)
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_location[i] /= scalar_products.sum()
    unit_location = np.nan_to_num(unit_location)

    return unit_location
