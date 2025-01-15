import numpy as np
from spikeinterface.core.numpyextractors import NumpySorting
from spikeinterface.core.sorting_tools import spike_vector_to_indices


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

    sorting = sorting_analyzer.sorting
    rng = np.random.RandomState(seed)
    fs = sorting_analyzer.sampling_frequency

    nb_splits = int(splitting_probability * len(sorting.unit_ids))
    if unit_ids is None:
        select_from = sorting.unit_ids
        if min_snr is not None:
            if sorting_analyzer.get_extension("noise_levels") is None:
                sorting_analyzer.compute("noise_levels")
            if sorting_analyzer.get_extension("quality_metrics") is None:
                sorting_analyzer.compute("quality_metrics", metric_names=["snr"])

            snr = sorting_analyzer.get_extension("quality_metrics").get_data()["snr"].values
            select_from = select_from[snr > min_snr]

        to_split_ids = rng.choice(select_from, nb_splits, replace=False)
    else:
        to_split_ids = unit_ids

    spikes = sorting_analyzer.sorting.to_spike_vector(concatenated=False)
    new_spikes = spikes[0].copy()
    max_index = np.max(new_spikes["unit_index"])
    new_unit_ids = list(sorting_analyzer.sorting.unit_ids.copy())
    spike_indices = spike_vector_to_indices(spikes, sorting_analyzer.unit_ids, absolute_index=True)
    splitted_pairs = []
    for unit_id in to_split_ids:
        ind_mask = spike_indices[0][unit_id]
        m = np.median(spikes[0][ind_mask]["sample_index"])
        time_mask = spikes[0][ind_mask]["sample_index"] > m
        mask = time_mask & (rng.rand(len(ind_mask)) <= partial_split_prob).astype(bool)
        new_index = int(unit_id) * np.ones(len(mask), dtype=bool)
        new_index[mask] = max_index + 1
        new_spikes["unit_index"][ind_mask] = new_index
        new_unit_ids += [max_index + 1]
        splitted_pairs += [(unit_id, new_unit_ids[-1])]
        max_index += 1

    new_sorting = NumpySorting(new_spikes, sampling_frequency=fs, unit_ids=new_unit_ids)
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

    if sorting_analyzer.get_extension("spike_amplitudes") is None:
        sorting_analyzer.compute("spike_amplitudes")

    rng = np.random.RandomState(seed)
    fs = sorting_analyzer.sampling_frequency
    from spikeinterface.core.template_tools import get_template_extremum_channel

    extremum_channel_inds = get_template_extremum_channel(sorting_analyzer, outputs="index")
    spikes = sorting_analyzer.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds, concatenated=False)
    new_spikes = spikes[0].copy()
    amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    nb_splits = int(splitting_probability * len(sorting_analyzer.sorting.unit_ids))

    if unit_ids is None:
        select_from = sorting_analyzer.sorting.unit_ids
        if min_snr is not None:
            if sorting_analyzer.get_extension("noise_levels") is None:
                sorting_analyzer.compute("noise_levels")
            if sorting_analyzer.get_extension("quality_metrics") is None:
                sorting_analyzer.compute("quality_metrics", metric_names=["snr"])

            snr = sorting_analyzer.get_extension("quality_metrics").get_data()["snr"].values
            select_from = select_from[snr > min_snr]
        to_split_ids = rng.choice(select_from, nb_splits, replace=False)
    else:
        to_split_ids = unit_ids

    max_index = np.max(new_spikes["unit_index"])
    new_unit_ids = list(sorting_analyzer.sorting.unit_ids.copy())
    splitted_pairs = []
    spike_indices = spike_vector_to_indices(spikes, sorting_analyzer.unit_ids, absolute_index=True)

    for unit_id in to_split_ids:
        ind_mask = spike_indices[0][unit_id]
        thresh = np.median(amplitudes[ind_mask])
        amplitude_mask = amplitudes[ind_mask] > thresh
        mask = amplitude_mask & (rng.rand(len(ind_mask)) <= partial_split_prob).astype(bool)
        new_index = int(unit_id) * np.ones(len(mask))
        new_index[mask] = max_index + 1
        new_spikes["unit_index"][ind_mask] = new_index
        new_unit_ids += [max_index + 1]
        splitted_pairs += [(unit_id, new_unit_ids[-1])]
        max_index += 1

    new_sorting = NumpySorting(new_spikes, sampling_frequency=fs, unit_ids=new_unit_ids)
    return new_sorting, splitted_pairs
