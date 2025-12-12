import numpy as np
from spikeinterface.metrics.spiketrain import (
    compute_firing_rates,
    compute_num_spikes,
)


def test_calculate_firing_ratess(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    firing_rates = compute_firing_rates(sorting_analyzer)
    assert np.all(np.array(list(firing_rates.values())) > 0)


def test_calculate_num_spikes(sorting_analyzer_simple):
    sorting_analyzer = sorting_analyzer_simple
    num_spikes = compute_num_spikes(sorting_analyzer)
    assert np.all(np.array(list(num_spikes.values())) > 0)
