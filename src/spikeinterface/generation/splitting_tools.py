def split_sorting_by_times(
    sorting_analyzer, splitting_probability=0.5, partial_split_prob=0.95, unit_ids=None, min_snr=None, seed=None
):
    sa = sorting_analyzer
    sorting = sa.sorting
    rng = np.random.RandomState(seed)

    sorting_split = sorting.select_units(sorting.unit_ids)
    split_units = []
    original_units = []
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

    import spikeinterface.curation as scur

    for unit in to_split_ids:
        num_spikes = len(sorting_split.get_unit_spike_train(unit))
        indices = np.zeros(num_spikes, dtype=int)
        indices[: num_spikes // 2] = (rng.rand(num_spikes // 2) < partial_split_prob).astype(int)
        indices[num_spikes // 2 :] = (rng.rand(num_spikes - num_spikes // 2) < 1 - partial_split_prob).astype(int)
        sorting_split = scur.split_unit_sorting(
            sorting_split, split_unit_id=unit, indices_list=indices, properties_policy="remove"
        )
        split_units.append(sorting_split.unit_ids[-2:])
        original_units.append(unit)
    return sorting_split, split_units


def split_sorting_by_amplitudes(sorting_analyzer, splitting_probability=0.5, unit_ids=None, min_snr=None, seed=None):
    """
    Fonction used to split a sorting based on the amplitudes of the units. This
    might be used for benchmarking meta merging step (see components)
    """

    sa = sorting_analyzer
    if sa.get_extension("spike_amplitudes") is None:
        sa.compute("spike_amplitudes")

    rng = np.random.RandomState(seed)

    from spikeinterface.core.numpyextractors import NumpySorting
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

        m = amplitudes[ind_mask].mean()
        s = amplitudes[ind_mask].std()
        thresh = m + 0.2 * s

        amplitude_mask = amplitudes > thresh
        mask = ind_mask & amplitude_mask
        new_spikes["unit_index"][mask] = max_index + 1

        amplitude_mask = (amplitudes > m) * (amplitudes < thresh)
        mask = ind_mask & amplitude_mask
        new_spikes["unit_index"][mask] = (max_index + 1) * rng.rand(np.sum(mask)) > 0.5
        max_index += 1
        new_unit_ids += [max(new_unit_ids) + 1]
        splitted_pairs += [(unit_id, new_unit_ids[-1])]

    new_sorting = NumpySorting(new_spikes, sampling_frequency=sa.sampling_frequency, unit_ids=new_unit_ids)
    return new_sorting, splitted_pairs