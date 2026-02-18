import numpy as np

from spikeinterface.core.analyzer_extension_core import BaseMetric
from spikeinterface.metrics.utils import compute_total_durations_per_unit


def compute_num_spikes(sorting_analyzer, unit_ids=None, periods=None):
    """
    Compute the number of spike across segments.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        The list of unit ids to compute the number of spikes. If None, all units are used.
    periods : array of unit_period_dtype, default: None
        Periods to consider for each unit.

    Returns
    -------
    num_spikes : dict
        The number of spikes, across all segments, for each unit ID.
    """
    sorting = sorting_analyzer.sorting
    sorting = sorting.select_periods(periods)
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    # re-order dict to match unit_ids order
    count_spikes = sorting.count_num_spikes_per_unit(unit_ids=unit_ids)
    num_spikes = {}
    for unit_id in unit_ids:
        num_spikes[unit_id] = count_spikes[unit_id]
    return num_spikes


class NumSpikes(BaseMetric):
    metric_name = "num_spikes"
    metric_function = compute_num_spikes
    metric_params = {}
    metric_descriptions = {"num_spikes": "Total number of spikes for each unit across all segments."}
    metric_columns = {"num_spikes": int}
    supports_periods = True


def compute_firing_rates(sorting_analyzer, unit_ids=None, periods=None):
    """
    Compute the firing rate across segments.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        The list of unit ids to compute the firing rate. If None, all units are used.
    periods : array of unit_period_dtype, default: None
        Periods to consider for each unit.

    Returns
    -------
    firing_rates : dict of floats
        The firing rate, across all segments, for each unit ID.
    """

    sorting = sorting_analyzer.sorting
    sorting = sorting.select_periods(periods)
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    total_durations = compute_total_durations_per_unit(sorting_analyzer, periods=periods)

    firing_rates = {}
    num_spikes = sorting.count_num_spikes_per_unit(unit_ids=unit_ids)
    for unit_id in unit_ids:
        if num_spikes[unit_id] == 0:
            firing_rates[unit_id] = np.nan
        else:
            firing_rates[unit_id] = num_spikes[unit_id] / total_durations[unit_id]
    return firing_rates


class FiringRate(BaseMetric):
    metric_name = "firing_rate"
    metric_function = compute_firing_rates
    metric_params = {}
    metric_descriptions = {"firing_rate": "Firing rate (spikes per second) for each unit across all segments."}
    metric_columns = {"firing_rate": float}
    supports_periods = True


spiketrain_metrics = [NumSpikes, FiringRate]
