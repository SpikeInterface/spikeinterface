import numpy as np
from spikeinterface.core.numpyextractors import NumpySorting


def split_sorting_by_times(
    sorting_analyzer, splitting_probability=0.5, partial_split_prob=0.95, unit_ids=None, min_snr=None, seed=None
):
    """
    Fonction used to split a sorting based on the times of the units. This
    might be used for benchmarking meta merging step (see components)

    Parameters
    ----------
    sorting_analyzer : A sortingAnalyzer object
        The sortingAnalyzer object whose sorting should be splitted
    splitting_probability : float, default 0.5
        probability of being splitted, for any cell in the provided sorting
    partial_split_prob : float, default 0.95
        The percentage of spikes that will belong to pre/post splits
    unit_ids : list of unit_ids, default None
        The list of unit_ids to be splitted, if prespecified
    min_snr : float, default=None
        If specified, only cells with a snr higher than min_snr might be splitted
    seed : int | None, default: None
        The seed for random generator.

    Returns
    -------
    new_sorting, splitted_pairs : The new splitted sorting, and the pairs that have been splitted
    """

    sa = sorting_analyzer
    sorting = sa.sorting
    rng = np.random.RandomState(seed)

    nb_splits = int(splitting_probability * len(sorting.unit_ids))
    if unit_ids is None:
        select_from = sorting.unit_ids
        if min_snr is not None:
            if sa.get_extension("noise_levels") is None:
                sa.compute("noise_levels")
            if sa.get_extension("quality_metrics") is None:
                sa.compute("quality_metrics", metric_names=["snr"])

            snr = sa.get_extension("quality_metrics").get_data()["snr"].values
            select_from = select_from[snr > min_snr]

        to_split_ids = rng.choice(select_from, nb_splits, replace=False)
    else:
        to_split_ids = unit_ids

    spikes = sa.sorting.to_spike_vector()
    new_spikes = spikes.copy()
    max_index = np.max(spikes["unit_index"])
    new_unit_ids = list(sa.sorting.unit_ids.copy())
    splitted_pairs = []
    for unit_id in to_split_ids:
        ind_mask = spikes["unit_index"] == sa.sorting.id_to_index(unit_id)
        m = np.median(spikes[ind_mask]["sample_index"])
        time_mask = spikes["sample_index"] > m
        mask = time_mask & (rng.rand(len(ind_mask)) <= partial_split_prob).astype(bool) & ind_mask
        new_spikes["unit_index"][mask] = max_index + 1
        new_unit_ids += [max_index + 1]
        splitted_pairs += [(unit_id, new_unit_ids[-1])]
        max_index += 1

    new_sorting = NumpySorting(new_spikes, sampling_frequency=sa.sampling_frequency, unit_ids=new_unit_ids)
    return new_sorting, splitted_pairs


def split_sorting_by_amplitudes(
    sorting_analyzer, splitting_probability=0.5, partial_split_prob=0.95, unit_ids=None, min_snr=None, seed=None
):
    """
    Fonction used to split a sorting based on the amplitudes of the units. This
    might be used for benchmarking meta merging step (see components)

    Parameters
    ----------
    sorting_analyzer : A sortingAnalyzer object
        The sortingAnalyzer object whose sorting should be splitted
    splitting_probability : float, default 0.5
        probability of being splitted, for any cell in the provided sorting
    partial_split_prob : float, default 0.95
        The percentage of spikes that will belong to pre/post splits
    unit_ids : list of unit_ids, default None
        The list of unit_ids to be splitted, if prespecified
    min_snr : float, default=None
        If specified, only cells with a snr higher than min_snr might be splitted
    seed : int | None, default: None
        The seed for random generator.

    Returns
    -------
    new_sorting, splitted_pairs : The new splitted sorting, and the pairs that have been splitted
    """

    sa = sorting_analyzer
    if sa.get_extension("spike_amplitudes") is None:
        sa.compute("spike_amplitudes")

    rng = np.random.RandomState(seed)
    from spikeinterface.core.template_tools import get_template_extremum_channel

    extremum_channel_inds = get_template_extremum_channel(sa, outputs="index")
    spikes = sa.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
    new_spikes = spikes.copy()
    amplitudes = sa.get_extension("spike_amplitudes").get_data()
    nb_splits = int(splitting_probability * len(sa.sorting.unit_ids))

    if unit_ids is None:
        select_from = sa.sorting.unit_ids
        if min_snr is not None:
            if sa.get_extension("noise_levels") is None:
                sa.compute("noise_levels")
            if sa.get_extension("quality_metrics") is None:
                sa.compute("quality_metrics", metric_names=["snr"])

            snr = sa.get_extension("quality_metrics").get_data()["snr"].values
            select_from = select_from[snr > min_snr]
        to_split_ids = rng.choice(select_from, nb_splits, replace=False)
    else:
        to_split_ids = unit_ids

    max_index = np.max(spikes["unit_index"])
    new_unit_ids = list(sa.sorting.unit_ids.copy())
    splitted_pairs = []
    for unit_id in to_split_ids:
        ind_mask = spikes["unit_index"] == sa.sorting.id_to_index(unit_id)
        thresh = np.median(amplitudes[ind_mask])
        amplitude_mask = amplitudes > thresh
        mask = amplitude_mask & (rng.rand(len(ind_mask)) <= partial_split_prob).astype(bool) & ind_mask
        new_spikes["unit_index"][mask] = max_index + 1
        new_unit_ids += [max_index + 1]
        splitted_pairs += [(unit_id, new_unit_ids[-1])]
        max_index += 1

    new_sorting = NumpySorting(new_spikes, sampling_frequency=sa.sampling_frequency, unit_ids=new_unit_ids)
    return new_sorting, splitted_pairs
