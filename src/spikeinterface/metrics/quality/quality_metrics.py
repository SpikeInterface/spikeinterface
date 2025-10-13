"""Classes and functions for computing multiple quality metrics."""

from __future__ import annotations

import numpy as np
from copy import deepcopy

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from spikeinterface.core.analyzer_extension_core import BaseMetricExtension

from .metric_classes import misc_metrics, pca_metrics

# from .quality_metric_list import (
#     compute_pc_metrics,
#     _misc_metric_name_to_func,
#     _possible_pc_metric_names,
#     qm_compute_name_to_column_names,
#     column_name_to_column_dtype,
#     metric_extension_dependencies,
# )
# from .misc_metrics_implementations import _default_params as misc_metrics_params
# from .pca_metrics_implementations import _default_params as pca_metrics_params


class ComputeQualityMetrics(BaseMetricExtension):
    """
    Compute quality metrics on a `sorting_analyzer`.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    metric_names : list or None
        List of quality metrics to compute.
    metric_params : dict of dicts or None
        Dictionary with parameters for quality metrics calculation.
        Default parameters can be obtained with: `si.qualitymetrics.get_default_qm_params()`
    skip_pc_metrics : bool, default: False
        If True, PC metrics computation is skipped.
    delete_existing_metrics : bool, default: False
        If True, any quality metrics attached to the `sorting_analyzer` are deleted. If False, any metrics which were previously calculated but are not included in `metric_names` are kept.

    Returns
    -------
    metrics: pandas.DataFrame
        Data frame with the computed metrics.

    Notes
    -----
    principal_components are loaded automatically if already computed.
    """

    extension_name = "quality_metrics"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = True
    need_backward_compatibility_on_load = True
    metric_list = misc_metrics + pca_metrics

    def _handle_backward_compatibility_on_load(self):
        # For backwards compatibility - this renames qm_params as metric_params
        if (qm_params := self.params.get("qm_params")) is not None:
            self.params["metric_params"] = qm_params
            del self.params["qm_params"]

    def _set_params(
        self,
        metric_names: list[str] | None = None,
        metric_params: dict | None = None,
        delete_existing_metrics: bool = False,
        # common extension kwargs
        peak_sign=None,
        seed=None,
        skip_pc_metrics=False,
    ):
        if metric_names is None:
            metric_names = [m.metric_name for m in self.metric_list]
            # if PC is available, PC metrics are automatically added to the list
            if skip_pc_metrics:
                pc_metric_names = [m.metric_name for m in pca_metrics]
                metric_names = [m for m in metric_names if m not in pc_metric_names]
            if "nn_advanced" in metric_names:
                # remove nn_advanced because too slow
                metric_names.remove("nn_advanced")

        return super()._set_params(
            metric_names=metric_names,
            metric_params=metric_params,
            delete_existing_metrics=delete_existing_metrics,
            peak_sign=peak_sign,
            seed=seed,
            skip_pc_metrics=skip_pc_metrics,
        )

    def _prepare_data(self, unit_ids=None):
        """Prepare shared data for quality metrics computation."""
        tmp_data = {}

        # Check if any PCA metrics are requested
        pca_metric_names = [m.metric_name for m in pca_metrics]
        requested_pca_metrics = [m for m in self.params["metric_names"] if m in pca_metric_names]

        if not requested_pca_metrics:
            return tmp_data

        # Check if PCA extension is available
        pca_ext = self.sorting_analyzer.get_extension("principal_components")
        if pca_ext is None:
            return tmp_data

        if unit_ids is None:
            unit_ids = self.sorting_analyzer.unit_ids

        # Pre-compute shared PCA data
        from spikeinterface.core.template_tools import get_template_extremum_channel
        from spikeinterface.metrics.quality.misc_metrics_implementations import compute_num_spikes, compute_firing_rates

        # Get dense PCA projections for all requested units
        dense_projections, spike_unit_indices = pca_ext.get_some_projections(channel_ids=None, unit_ids=unit_ids)
        all_labels = self.sorting_analyzer.sorting.unit_ids[spike_unit_indices]

        # Get extremum channels for neighbor selection in sparse mode
        extremum_channels = get_template_extremum_channel(self.sorting_analyzer)

        # tmp_data["dense_projections"] = dense_projections
        # tmp_data["spike_unit_indices"] = spike_unit_indices
        # tmp_data["all_labels"] = all_labels
        # tmp_data["extremum_channels"] = extremum_channels
        # tmp_data["pca_mode"] = pca_ext.params["mode"]
        # tmp_data["channel_ids"] = self.sorting_analyzer.channel_ids

        # Pre-compute spike counts and firing rates if advanced NN metrics are requested
        advanced_nn_metrics = ["nn_advanced"]  # Our grouped advanced NN metric
        if any(m in advanced_nn_metrics for m in requested_pca_metrics):
            tmp_data["n_spikes_all_units"] = compute_num_spikes(self.sorting_analyzer, unit_ids=unit_ids)
            tmp_data["fr_all_units"] = compute_firing_rates(self.sorting_analyzer, unit_ids=unit_ids)

        # Pre-compute per-unit PCA data and neighbor information
        pca_data_per_unit = {}
        for unit_id in unit_ids:
            # Determine neighbor units based on sparsity
            if self.sorting_analyzer.is_sparse():
                neighbor_channel_ids = self.sorting_analyzer.sparsity.unit_id_to_channel_ids[unit_id]
                neighbor_unit_ids = [
                    other_unit for other_unit in unit_ids if extremum_channels[other_unit] in neighbor_channel_ids
                ]
                neighbor_channel_indices = self.sorting_analyzer.channel_ids_to_indices(neighbor_channel_ids)
            else:
                neighbor_channel_ids = self.sorting_analyzer.channel_ids
                neighbor_unit_ids = unit_ids
                neighbor_channel_indices = self.sorting_analyzer.channel_ids_to_indices(neighbor_channel_ids)

            # Filter projections to neighbor units
            labels = all_labels[np.isin(all_labels, neighbor_unit_ids)]
            if pca_ext.params["mode"] == "concatenated":
                pcs = dense_projections[np.isin(all_labels, neighbor_unit_ids)]
            else:
                pcs = dense_projections[np.isin(all_labels, neighbor_unit_ids)][:, :, neighbor_channel_indices]
            pcs_flat = pcs.reshape(pcs.shape[0], -1)

            pca_data_per_unit[unit_id] = {
                "pcs_flat": pcs_flat,
                "labels": labels,
                "neighbor_unit_ids": neighbor_unit_ids,
                "neighbor_channel_ids": neighbor_channel_ids,
                "neighbor_channel_indices": neighbor_channel_indices,
            }

        tmp_data["pca_data_per_unit"] = pca_data_per_unit

        return tmp_data


register_result_extension(ComputeQualityMetrics)
compute_quality_metrics = ComputeQualityMetrics.function_factory()


def get_quality_metric_list():
    """
    Return a list of the available quality metrics.
    """

    return [m.metric_name for m in ComputeQualityMetrics.metric_list]


def get_quality_pca_metric_list():
    """
    Return a list of the available quality PCA metrics.
    """

    return [m.metric_name for m in pca_metrics]


def get_default_qm_params(metric_names=None):
    """
    Return default dictionary of quality metrics parameters.

    Returns
    -------
    dict
        Default qm parameters with metric name as key and parameter dictionary as values.
    """
    default_params = ComputeQualityMetrics.get_default_metric_params()
    if metric_names is None:
        return default_params
    else:
        metric_names = list(set(metric_names) & set(default_params.keys()))
        metric_params = {m: default_params[m] for m in metric_names}
        return metric_params
