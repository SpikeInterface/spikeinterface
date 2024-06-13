from __future__ import annotations
import numpy as np

def presence_distance(sorting, unit1, unit2, bin_duration_s=2, bins=None):
    """
    Compute the presence distance between two units.

    The presence distance is defined as the Wasserstein distance between the two histograms of
    the firing activity over time.

    Parameters
    ----------
    sorting: Sorting
        The sorting object.
    unit1: int or str
        The id of the first unit.
    unit2: int or str
        The id of the second unit.
    bin_duration_s: float
        The duration of the bin in seconds.
    bins: array-like
        The bins used to compute the firing rate.

    Returns
    -------
    d: float
        The presence distance between the two units.
    """
    if bins is None:
        bin_size = bin_duration_s * sorting.sampling_frequency
        bins = np.arange(0, sorting.get_num_samples(), bin_size)

    st1 = sorting.get_unit_spike_train(unit_id=unit1)
    st2 = sorting.get_unit_spike_train(unit_id=unit2)

    h1, _ = np.histogram(st1, bins)
    h1 = h1.astype(float)

    h2, _ = np.histogram(st2, bins)
    h2 = h2.astype(float)

    import scipy

    xaxis = bins[1:] / sorting.sampling_frequency
    d = scipy.stats.wasserstein_distance(xaxis, xaxis, h1, h2)

    return d


def compute_presence_distance(sorting, pair_mask, **presence_distance_kwargs):
    """
    Get the potential drift-related merges based on similarity and presence completeness.

    Parameters
    ----------
    sorting: Sorting
        The sorting object
    pair_mask: None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.
    presence_distance_threshold: float
        The presence distance threshold used to consider two units as similar
    presence_distance_kwargs: A dictionary of kwargs to be passed to compute_presence_distance()

    Returns
    -------
    potential_merges: list
        The list of potential merges

    """

    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype="bool")

    presence_distances = np.ones((sorting.get_num_units(), sorting.get_num_units()))

    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue
            unit1 = unit_ids[unit_ind1]
            unit2 = unit_ids[unit_ind2]
            d = presence_distance(sorting, unit1, unit2, **presence_distance_kwargs)
            presence_distances[unit_ind1, unit_ind2] = d

    return presence_distances
