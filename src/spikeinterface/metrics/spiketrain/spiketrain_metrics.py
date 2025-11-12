from __future__ import annotations

from spikeinterface.core.sortinganalyzer import register_result_extension
from spikeinterface.core.analyzer_extension_core import BaseMetricExtension

from .metrics import spiketrain_metrics


class ComputeSpikeTrainMetrics(BaseMetricExtension):
    """
    Compute spike train metrics including:
        * num_spikes
        * firing_rate
        * TODO: add ACG/ISI metrics
        * TODO: add burst metrics

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    metric_names : list or None, default: None
        List of metrics to compute (see si.metrics.get_spiketrain_metric_names())
    delete_existing_metrics : bool, default: False
        If True, any template metrics attached to the `sorting_analyzer` are deleted. If False, any metrics which were previously calculated but are not included in `metric_names` are kept, provided the `metric_params` are unchanged.
    metric_params : dict of dicts or None, default: None
        Dictionary with parameters for template metrics calculation.
        Default parameters can be obtained with: `si.metrics.get_default_spiketrain_metrics_params()`

    Returns
    -------
    spiketrain_metrics : pd.DataFrame
        Dataframe with the computed spike train metrics.

    Notes
    -----
    If any multi-channel metric is in the metric_names or include_multi_channel_metrics is True, sparsity must be None,
    so that one metric value will be computed per unit.
    For multi-channel metrics, 3D channel locations are not supported. By default, the depth direction is "y".
    """

    extension_name = "spiketrain_metrics"
    depend_on = []
    need_backward_compatibility_on_load = True
    metric_list = spiketrain_metrics


register_result_extension(ComputeSpikeTrainMetrics)
compute_spiketrain_metrics = ComputeSpikeTrainMetrics.function_factory()


def get_spiketrain_metric_list():
    return [m.metric_name for m in spiketrain_metrics]


def get_default_spiketrain_metrics_params(metric_names=None):
    default_params = ComputeSpikeTrainMetrics.get_default_metric_params()
    if metric_names is None:
        return default_params
    else:
        metric_names = list(set(metric_names) & set(default_params.keys()))
        metric_params = {m: default_params[m] for m in metric_names}
        return metric_params
