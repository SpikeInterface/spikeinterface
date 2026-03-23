import numpy as np

from spikeinterface.core.base import unit_period_dtype
from spikeinterface.metrics.utils import create_regular_periods
from spikeinterface.metrics.spiketrain.metrics import compute_num_spikes, compute_firing_rates


def test_calculate_num_spikes(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    # spike_amps = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    num_spikes = compute_num_spikes(sorting_analyzer)
    periods = create_regular_periods(sorting_analyzer, num_periods=5)
    num_spikes_periods = compute_num_spikes(sorting_analyzer, periods=periods)
    assert num_spikes == num_spikes_periods

    # calculate num spikes with empty periods
    empty_periods = np.empty(0, dtype=unit_period_dtype)
    num_spikes_empty_periods = compute_num_spikes(sorting_analyzer, periods=empty_periods)
    assert num_spikes_empty_periods == {unit_id: 0 for unit_id in sorting_analyzer.sorting.unit_ids}


def test_calculate_firing_rates(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    # spike_amps = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    firing_rates = compute_firing_rates(sorting_analyzer)
    periods = create_regular_periods(sorting_analyzer, num_periods=5)
    firing_rates_periods = compute_firing_rates(sorting_analyzer, periods=periods)
    assert firing_rates == firing_rates_periods

    # calculate num spikes with empty periods
    empty_periods = np.empty(0, dtype=unit_period_dtype)
    firing_rates_empty_periods = compute_firing_rates(sorting_analyzer, periods=empty_periods)
    assert np.all(np.isnan(np.array(list(firing_rates_empty_periods.values()))))
