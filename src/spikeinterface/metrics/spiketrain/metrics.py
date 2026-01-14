import numpy as np
from spikeinterface.core.analyzer_extension_core import BaseMetric


def compute_num_spikes(sorting_analyzer, unit_ids=None, **kwargs):
    """
    Compute the number of spike across segments.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        The list of unit ids to compute the number of spikes. If None, all units are used.

    Returns
    -------
    num_spikes : dict
        The number of spikes, across all segments, for each unit ID.
    """

    sorting = sorting_analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    num_segs = sorting.get_num_segments()

    num_spikes = {}
    for unit_id in unit_ids:
        n = 0
        for segment_index in range(num_segs):
            st = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            n += st.size
        num_spikes[unit_id] = n

    return num_spikes


class NumSpikes(BaseMetric):
    metric_name = "num_spikes"
    metric_function = compute_num_spikes
    metric_params = {}
    metric_descriptions = {"num_spikes": "Total number of spikes for each unit across all segments."}
    metric_columns = {"num_spikes": int}


def compute_firing_rates(sorting_analyzer, unit_ids=None):
    """
    Compute the firing rate across segments.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    unit_ids : list or None
        The list of unit ids to compute the firing rate. If None, all units are used.

    Returns
    -------
    firing_rates : dict of floats
        The firing rate, across all segments, for each unit ID.
    """

    sorting = sorting_analyzer.sorting
    if unit_ids is None:
        unit_ids = sorting.unit_ids
    total_duration = sorting_analyzer.get_total_duration()

    firing_rates = {}
    num_spikes = compute_num_spikes(sorting_analyzer)
    for unit_id in unit_ids:
        if num_spikes[unit_id] == 0:
            firing_rates[unit_id] = np.nan
        else:
            firing_rates[unit_id] = num_spikes[unit_id] / total_duration
    return firing_rates


class FiringRate(BaseMetric):
    metric_name = "firing_rate"
    metric_function = compute_firing_rates
    metric_params = {}
    metric_descriptions = {"firing_rate": "Firing rate (spikes per second) for each unit across all segments."}
    metric_columns = {"firing_rate": float}


spiketrain_metrics = [NumSpikes, FiringRate]
