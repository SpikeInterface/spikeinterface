from pathlib import Path
from spikeinterface.core import read_python
import numpy as np
import pandas as pd


def create_ks_drift_map(sorter_output: str | Path, localised_spikes_only: bool = False):
    """ """
    params = load_ks_dir(sorter_output, load_pcs=True, exclude_noise=False)

    localised_templates = []
    if localised_spikes_only:

        for idx, template in enumerate(params["templates"]):
            max_channel = np.max(np.abs(params["templates"][idx, :, :]))
            channels_over_threshold = (
                np.max(np.abs(params["templates"][idx, :, :]), axis=0) > 0.5 * max_channel
            )  #     np.max(np.max(np.abs(params["templates"]), axis=1), axis=1)
            channel_ids_over_threshold = np.where(channels_over_threshold)[0]

            if np.max(channel_ids_over_threshold) - np.min(channels_over_threshold) <= 20:
                localised_templates.append(idx)

        localised_template_by_spike = np.isin(params["spike_templates"].squeeze(), localised_templates)

        params["spike_templates"] = params["spike_templates"][localised_template_by_spike]
        params["spike_times"] = params["spike_times"][localised_template_by_spike]
        params["spike_clusters"] = params["spike_clusters"][localised_template_by_spike]
        params["temp_scaling_amplitudes"] = params["temp_scaling_amplitudes"][localised_template_by_spike]
        params["pc_features"] = params["pc_features"][localised_template_by_spike]

    pc_features = params["pc_features"][:, 0, :]
    pc_features[pc_features < 0] = 0

    # KS stores spikes in a low-channel representation. e.g. if 384 channels, nearest 32
    # channels are taken. The ids of these channels are in pc_features_indices. The SVD
    # of the 32 channels is taken, three top PC stored. For each channel loading onto
    # top 3 PC is stored. This is super confusing if expecting full channel number.
    # For example in my data 384 channels but nPCFeatures is 32. How is spatial information
    # thus incorporated?
    spike_features_indices = params["pc_features_indices"][params["spike_templates"], :].squeeze()

    ycoords = params["channel_positions"][:, 1]
    spike_feature_ycoords = ycoords[spike_features_indices]

    spike_depths = np.sum(spike_feature_ycoords * pc_features**2, axis=1) / np.sum(
        pc_features**2, axis=1
    )  # TODO: better document, write this out somewhere

    # spike_amplitudes, spike_depths, template_depths, template_amplitudes, unwhite_templates, trough_peak_durations, waveforms
    spike_amplitudes, _, _, _, unwhite_templates, _, _ = (
        template_positions_amplitudes(  # TODO: anything to do about this?
            params["templates"],
            params["whitening_matrix_inv"],
            ycoords,
            params["spike_templates"],
            params["temp_scaling_amplitudes"],
        )
    )


# for plotting, we need the amplitude of each spike, both so we can draw a
# threshold and exclude the low-amp noisy ones, and so we can color the
# points by the amplitude


def template_positions_amplitudes(
    templates, inverse_whitening_matrix, ycoords, spike_templates, template_scaling_amplitudes
):
    """ """
    spike_templates = spike_templates.squeeze()

    unwhite_templates = np.zeros_like(templates)

    for idx, template in enumerate(templates):
        unwhite_templates[idx, :, :] = templates[idx, :, :] @ inverse_whitening_matrix

    template_channel_amplitudes = np.max(unwhite_templates, axis=1) - np.min(unwhite_templates, axis=1)

    template_amplitudes_unscaled = np.max(template_channel_amplitudes, axis=1)

    threshold_values = 0.3 * template_amplitudes_unscaled

    template_channel_amplitudes[template_channel_amplitudes < threshold_values[:, np.newaxis]] = 0

    # weight by amplitude here, before we weight by PC loading
    template_depths = np.sum(template_channel_amplitudes * ycoords[np.newaxis, :], axis=1) / np.sum(
        template_channel_amplitudes, axis=1
    )
    spike_amplitudes = (
        template_amplitudes_unscaled[spike_templates] * template_scaling_amplitudes.squeeze()
    )  # TODO: handle these squeezes

    # take the average of all spike amps to get actual template amps (since
    # tempScalingAmps are equal mean for all templates)
    # TOOD: test carefully, 99% sure it is doing the  same thing here
    num_indices = templates.shape[0]
    sum_per_index = np.zeros(num_indices)
    np.add.at(sum_per_index, spike_templates, spike_amplitudes)
    counts = np.bincount(spike_templates)
    template_amplitudes = np.divide(sum_per_index, counts, out=np.zeros_like(sum_per_index), where=counts != 0)

    # Each spike's depth is the depth of its template
    spike_depths = template_depths[spike_templates]

    # Get channel with the largest amplitude, take that as the waveform
    max_site = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    templates_max = np.empty(templates.shape[:2])
    templates_max.fill(np.nan)

    for idx, template in enumerate(templates):
        templates_max[idx, :] = templates[idx, :, max_site[idx]]

    waveforms = templates_max  # won't copy this, but make sure never to edit in place before end of function.

    # Get trough-to-peak time for each template
    waveform_trough = np.argmin(templates_max, axis=1)

    trough_peak_durations = np.zeros(templates_max.shape[0])  # TOOD: num templates var?
    for idx, template_max in enumerate(templates_max):
        trough_peak_durations[idx] = np.argmax(template_max[waveform_trough[idx] :])

    return (
        spike_amplitudes,
        spike_depths,
        template_depths,
        template_amplitudes,
        unwhite_templates,
        trough_peak_durations,
        waveforms,
    )  # TODO: check carefully against matlab


def load_ks_dir(sorter_output: Path, exclude_noise: bool = True, load_pcs: bool = False) -> dict:
    """ """
    sorter_output = Path(sorter_output)

    params = read_python(sorter_output / "params.py")

    spike_times = np.load(sorter_output / "spike_times.npy") / params["sample_rate"]
    spike_templates = np.load(sorter_output / "spike_templates.npy")

    if (clusters_path := sorter_output / "spike_clusters.csv").is_dir():
        spike_clusters = np.load(clusters_path)
    else:
        spike_clusters = spike_templates.copy()

    temp_scaling_amplitudes = np.load(sorter_output / "amplitudes.npy")

    if load_pcs:
        pc_features = np.load(sorter_output / "pc_features.npy")
        pc_features_indices = np.load(sorter_output / "pc_feature_ind.npy")
    else:
        pc_features = pc_features_indices = None

    # This makes the assumption that there will never be different .csv and .tsv files
    # in the same sorter output (this should never happen, there will never even be two).
    # Though can be saved as .tsv, it seems the .csv is also tab formatted as far as pandas is concerned.
    # TODO: .tsv suffix is untested.
    if exclude_noise and (
        (cluster_path := sorter_output / "cluster_groups.csv").is_file()
        or (cluster_path := sorter_output / "cluster_group.tsv").is_file()
    ):
        # TODO: need to check a) why there can be cluster id missing
        #  from cluster_groups and b) this handles the case correctly
        # TODO: test against csv
        cluster_ids, cluster_groups = load_cluster_groups(cluster_path)

        noise_cluster_ids = cluster_ids[cluster_groups == 0]
        not_noise_clusters_by_spike = ~np.isin(spike_clusters.ravel(), noise_cluster_ids)

        spike_times = spike_times[not_noise_clusters_by_spike]
        spike_templates = spike_templates[not_noise_clusters_by_spike]
        temp_scaling_amplitudes = temp_scaling_amplitudes[not_noise_clusters_by_spike]

        if load_pcs:
            pc_features = pc_features[not_noise_clusters_by_spike, :, :]

        spike_clusters = spike_clusters[not_noise_clusters_by_spike]
        cluster_ids = cluster_ids[cluster_groups != 0]
        cluster_groups = cluster_groups[cluster_groups != 0]
    else:
        cluster_ids = np.unique(spike_clusters)
        cluster_groups = 3 * np.ones(cluster_ids.size)

    new_params = {
        "spike_times": spike_times,
        "spike_templates": spike_templates,
        "spike_clusters": spike_clusters,
        "pc_features": pc_features,
        "pc_features_indices": pc_features_indices,
        "temp_scaling_amplitudes": temp_scaling_amplitudes,
        "cluster_ids": cluster_ids,
        "cluster_groups": cluster_groups,
        "channel_positions": np.load(sorter_output / "channel_positions.npy"),
        "templates": np.load(sorter_output / "templates.npy"),
        "whitening_matrix_inv": np.load(sorter_output / "whitening_mat_inv.npy"),
    }

    params.update(new_params)

    return params


def load_cluster_groups(cluster_path):
    """"""
    cluster_groups_table = pd.read_csv(cluster_path, sep="\t")

    group_key = cluster_groups_table.columns[1]  # "groups" (csv) or "KSLabel" (tsv)

    for key, _id in zip(
        ["noise", "mua", "good", "unsorted"],
        ["0", "1", "2", "3"],  # required as str to avoid pandas replace downcast FutureWarning
    ):
        cluster_groups_table[group_key] = cluster_groups_table[group_key].replace(key, _id)

    cluster_ids = cluster_groups_table["cluster_id"].to_numpy()
    cluster_groups = cluster_groups_table[group_key].astype(int).to_numpy()

    return cluster_ids, cluster_groups


create_ks_drift_map("/Users/joeziminski/data/bombcelll/sorter_output", localised_spikes_only=False)
