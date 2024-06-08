from __future__ import annotations
import numpy as np

from .auto_merge import check_improve_contaminations_score, compute_templates_diff, compute_refrac_period_violations


def presence_distance(sorting, unit1, unit2, bin_duration_s=2, percentile_norm=90, bins=None):
    """
    Compute the presence distance between two units.

    The presence distance is defined as the sum of the absolute difference between the sum of
    the normalized firing profiles of the two units and a constant firing profile.

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
    percentile_norm: float
        The percentile used to normalize the firing rate.
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
    norm_value1 = np.percentile(h1, percentile_norm)

    h2, _ = np.histogram(st2, bins)
    h2 = h2.astype(float)
    norm_value2 = np.percentile(h2, percentile_norm)

    if not np.isnan(norm_value1) and not np.isnan(norm_value2) and norm_value1 > 0 and norm_value2 > 0:
        h1 = h1 / norm_value1
        h2 = h2 / norm_value2
        d = np.sum(np.abs(h1 + h2 - np.ones_like(h1))) / sorting.get_total_duration()
    else:
        d = 1.0

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


def get_potential_temporal_splits(
    sorting_analyzer,
    minimum_spikes=100,
    presence_distance_threshold=0.1,
    template_diff_thresh=0.25,
    censored_period_ms=0.3,
    refractory_period_ms=1.0,
    num_channels=5,
    num_shift=5,
    contamination_threshold=0.2,
    firing_contamination_balance=1.5,
    extra_outputs=False,
    steps=None,
    template_metric="l1",
    **presence_distance_kwargs,
):
    """
    Algorithm to find and check potential temporal merges between units.

    The merges are proposed when the following criteria are met:

        * STEP 1: enough spikes are found in each units for computing the correlogram (`minimum_spikes`)
        * STEP 2: the templates of the two units are similar (`template_diff_thresh`)
        * STEP 3: the presence distance of the two units is high
        * STEP 4: the unit "quality score" is increased after the merge.

    The "quality score" factors in the increase in firing rate (**f**) due to the merge and a possible increase in
    contamination (**C**), wheighted by a factor **k** (`firing_contamination_balance`).

    .. math::

        Q = f(1 - (k + 1)C)


    """

    import scipy

    sorting = sorting_analyzer.sorting
    recording = sorting_analyzer.recording
    unit_ids = sorting.unit_ids
    sorting.register_recording(recording)

    # to get fast computation we will not analyse pairs when:
    #    * not enough spikes for one of theses
    #    * auto correlogram is contaminated
    #    * to far away one from each other

    if steps is None:
        steps = [
            "min_spikes",
            "remove_contaminated",
            "template_similarity",
            "presence_distance",
            "check_increase_score",
        ]

    n = unit_ids.size
    pair_mask = np.ones((n, n), dtype="bool")

    # STEP 1 :
    if "min_spikes" in steps:
        num_spikes = sorting.count_num_spikes_per_unit(outputs="array")
        to_remove = num_spikes < minimum_spikes
        pair_mask[to_remove, :] = False
        pair_mask[:, to_remove] = False

    # STEP 2 : remove contaminated auto corr
    if "remove_contaminated" in steps:
        contaminations, nb_violations = compute_refrac_period_violations(
            sorting_analyzer, refractory_period_ms=refractory_period_ms, censored_period_ms=censored_period_ms
        )
        nb_violations = np.array(list(nb_violations.values()))
        contaminations = np.array(list(contaminations.values()))
        to_remove = contaminations > contamination_threshold
        pair_mask[to_remove, :] = False
        pair_mask[:, to_remove] = False

    # STEP 2 : check if potential merge with CC also have template similarity
    if "template_similarity" in steps:
        templates_ext = sorting_analyzer.get_extension("templates")
        assert (
            templates_ext is not None
        ), "auto_merge with template_similarity requires a SortingAnalyzer with extension templates"

        template_similarity_ext = sorting_analyzer.get_extension("template_similarity")
        if template_similarity_ext is not None:
            templates_diff = template_similarity_ext.get_data()
        else:
            templates_array = templates_ext.get_data(outputs="numpy")

            templates_diff = compute_templates_diff(
                sorting,
                templates_array,
                num_channels=num_channels,
                num_shift=num_shift,
                pair_mask=pair_mask,
                template_metric=template_metric,
                sparsity=sorting_analyzer.sparsity,
            )

        pair_mask = pair_mask & (templates_diff < template_diff_thresh)

    # STEP 3 : validate the potential merges with CC increase the contamination quality metrics
    if "presence_distance" in steps:
        presence_distances = compute_presence_distance(sorting, pair_mask, **presence_distance_kwargs)
        pair_mask = pair_mask & (presence_distances < presence_distance_threshold)

    # STEP 4 : validate the potential merges with CC increase the contamination quality metrics
    if "check_increase_score" in steps:
        pair_mask, pairs_decreased_score = check_improve_contaminations_score(
            sorting_analyzer,
            pair_mask,
            contaminations,
            firing_contamination_balance,
            refractory_period_ms,
            censored_period_ms,
        )

    # FINAL STEP : create the final list from pair_mask boolean matrix
    ind1, ind2 = np.nonzero(pair_mask)
    potential_merges = list(zip(unit_ids[ind1], unit_ids[ind2]))

    if extra_outputs:
        outs = dict(
            templates_diff=templates_diff,
            presence_distances=presence_distances,
            pairs_decreased_score=pairs_decreased_score,
        )
        return potential_merges, outs
    else:
        return potential_merges
