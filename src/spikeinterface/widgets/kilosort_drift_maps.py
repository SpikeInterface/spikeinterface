from pathlib import Path

import scipy.signal
from spikeinterface.core import read_python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats


# TODO: unit test localised spikes only
# unit test and document load_cluster_groups
# unit test and document load_ks_dir
# unit test and document template_positions_amplitudes


def plot_ks_drift_map(
    sorter_output: str | Path,
    localised_spikes_only: bool = False,
    only_include_large_amplitude_spikes=True,
    decimate=False,
    add_histogram_plot=True,
    weight_histogram_by_amplitude=False,
    add_histogram_peaks_and_boundaries=True,
    add_drift_events=True,
    gain: float | None = None,
):
    spike_times, spike_amplitudes, spike_depths, _ = create_ks_drift_map(sorter_output, localised_spikes_only, gain)

    # Do this first, so we always at the same scale no matter what
    # decimation or filtering by amplitude we do.
    amplitude_range_all_spikes = (
        spike_amplitudes.min(),
        spike_amplitudes.max(),
    )

    if decimate:
        spike_times = spike_times[::decimate]
        spike_amplitudes = spike_amplitudes[::decimate]
        spike_depths = spike_depths[::decimate]

    # break up into 800um segments for (arbitary) amplitude values
    if only_include_large_amplitude_spikes:
        spike_times, spike_depths, spike_amplitudes = filter_large_amplitude_spikes(
            spike_times, spike_depths, spike_amplitudes
        )

    fig = plt.figure(figsize=(10, 10 * (6 / 8)))

    if add_histogram_plot:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
    else:
        ax2 = fig.add_subplot()  # TODO: rename axis

    plot_kilosort_drift_map_raster(
        spike_times,
        spike_amplitudes,
        spike_depths,
        amplitude_range_all_spikes,
        axis=ax2,
    )

    if not add_histogram_plot:
        ax2.set_xlabel("time")
        ax2.set_ylabel("y position")
        plt.show()
        return

    ax1.set_xlabel("count")
    ax2.set_xlabel("time")
    ax1.set_ylabel("y position")

    # Plot histogram on the left edge
    bin_um = 2
    bins = np.arange(spike_depths.min() - bin_um, spike_depths.max() + bin_um, bin_um)
    counts, bins = np.histogram(spike_depths, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if weight_histogram_by_amplitude:
        bin_indices = np.digitize(spike_depths, bins, right=True) - 1
        counts = np.zeros(bin_indices.max() + 1)
        spike_amplitudes /= spike_amplitudes.max()
        np.add.at(counts, bin_indices, spike_amplitudes)

    ax1.plot(counts, bin_centers, color="black", linewidth=1)

    if add_histogram_peaks_and_boundaries:
        color_histogram_peaks_and_detect_drift_events(
            spike_times, spike_depths, counts, bin_centers, ax1, ax2, add_drift_events
        )
    plt.show()


def color_histogram_peaks_and_detect_drift_events(
    spike_times, spike_depths, counts, bin_centers, ax1, ax2, add_drift_events
):

    all_peak_indexes = scipy.signal.find_peaks(
        counts,
    )[0]

    # new step to filter low-frequency peaks so they are not included in the
    # step to determine whether peaks are overlapping.
    filtered_peak_indexes = []
    for peak_idx in all_peak_indexes:
        if counts[peak_idx] > 0.25 * spike_times[-1]:
            filtered_peak_indexes.append(peak_idx)
    filtered_peak_indexes = np.array(filtered_peak_indexes)

    for idx, peak_index in enumerate(filtered_peak_indexes):

        peak_count = counts[peak_index]

        #  we want the peaks to correspond to some minimal firing rate
        #  (otherwise peaks by very few spikes will be considered as well...)
        start_position = np.where(counts[:peak_index] < peak_count * 0.05)[0].max()
        end_position = np.where(counts[peak_index:] < peak_count * 0.05)[0].min() + peak_index

        if (
            idx > 0
            and start_position < filtered_peak_indexes[idx - 1]  # TODO: THINK
            or idx < filtered_peak_indexes.size - 1
            and end_position > filtered_peak_indexes[idx + 1]
        ):

            ax1.scatter(peak_count, bin_centers[peak_index], facecolors="none", edgecolors="blue")
            continue

        else:
            for position in [start_position, end_position]:
                ax1.axhline(bin_centers[position], 0, counts.max(), color="grey", linestyle="--")
            ax1.scatter(peak_count, bin_centers[peak_index], facecolors="none", edgecolors="red")

        # detect drift events
        if add_drift_events:
            I = np.logical_and(
                spike_depths > bin_centers[start_position],
                spike_depths < bin_centers[end_position],
            )
            current_spike_depths = spike_depths[I].squeeze()
            current_spike_times = spike_times[I].squeeze()

            window_s = 10

            num_bins = np.round(spike_times[-1].squeeze() / window_s).astype(int)
            drift_events = []
            x = np.linspace(0, spike_times[-1].squeeze(), num_bins)
            x = np.arange(0, np.ceil(spike_times[-1]).astype(int), 10)
            for t in x:  # TODO: rename

                I = np.logical_and(current_spike_times >= t, current_spike_times <= t + window_s)
                drift_size = bin_centers[peak_index] - np.median(current_spike_depths[I])

                # 6 um is the hardcoded threshold for drift, and we want at least 10 spikes for the median calculation

                if np.abs(drift_size) > 6 and I[I].size > 10:
                    drift_events.append((t + 5, bin_centers[peak_index], drift_size))
            drift_events = np.array(drift_events)

            if np.any(drift_events):
                ax2.scatter(drift_events[:, 0], drift_events[:, 1], facecolors="r", edgecolors="none")
                for i, _ in enumerate(drift_events):
                    ax2.text(drift_events[i, 0] + 1, drift_events[i, 1], str(round(drift_events[i, 2])), color="r")


def plot_kilosort_drift_map_raster(spike_times, spike_amplitudes, spike_depths, amplitude_range, axis):
    """ """
    n_color_bins = 20
    marker_size = 0.5

    color_bins = np.linspace(amplitude_range[0], amplitude_range[1], n_color_bins)

    # Create a grayscale colormap and reverse it
    colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]

    # TODO: rewrite
    for b in range(n_color_bins - 1):
        these_spikes = (spike_amplitudes >= color_bins[b]) & (spike_amplitudes <= color_bins[b + 1])

        axis.scatter(
            spike_times[these_spikes], spike_depths[these_spikes], color=colors[b], s=marker_size, antialiased=True
        )


def filter_large_amplitude_spikes(spike_times, spike_depths, spike_amplitudes):
    """ """
    spike_bool = np.zeros_like(spike_amplitudes, dtype=bool)

    segment_size_um = 800
    probe_segments = np.arange(np.floor(spike_depths.max() / segment_size_um) + 1) * segment_size_um

    for segment_left_edge in probe_segments:
        spikes_in_seg = np.where(
            np.logical_and(spike_depths >= segment_left_edge, spike_depths < segment_left_edge + segment_size_um)
        )[0]
        spike_amps = spike_amplitudes[spikes_in_seg]
        I = spike_amps > np.mean(spike_amps) + 1.5 * np.std(spike_amps, ddof=1)

        spike_bool[spikes_in_seg] = I

    spike_times = spike_times[spike_bool]
    spike_depths = spike_depths[spike_bool]
    spike_amplitudes = spike_amplitudes[spike_bool]

    return spike_times, spike_depths, spike_amplitudes


def create_ks_drift_map(sorter_output: str | Path, localised_spikes_only: bool = False, gain: float | None = None):
    """ """
    params = load_ks_dir(sorter_output, load_pcs=True, exclude_noise=False)

    if localised_spikes_only:
        # TODO: this section is not tested against matlab
        localised_templates = []

        for idx, template in enumerate(params["templates"]):
            max_channel = np.max(np.abs(params["templates"][idx, :, :]))
            channels_over_threshold = np.max(np.abs(params["templates"][idx, :, :]), axis=0) > 0.5 * max_channel
            channel_ids_over_threshold = np.where(channels_over_threshold)[0]

            if np.max(channel_ids_over_threshold) - np.min(channel_ids_over_threshold) <= 20:
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

    # TODO: isn't this calculated already in the subfunction?
    max_site = np.argmax(
        np.max(np.abs(unwhite_templates), axis=1), axis=1
    )  # TODO: but didn't we already calculate the depths?
    spike_sites = max_site[params["spike_templates"].squeeze()]  # TODO: what to do about array squeeze...

    if gain is not None:
        spike_amplitudes *= gain

    # Above we already removed non-localized templates, but that on its own is insufficient.
    # Note for IMEC probe adding a constant term kills the regression making the regressors rank deficient

    # TOOD: this regression is not identical because spike_depths is not
    # identical because of numerical differences between MATLAB and python.

    if localised_spikes_only:
        # Interpolate the channel ids to location. Remove spikes > 5 um from
        # average position
        b = stats.linregress(spike_depths, spike_sites).slope
        i = np.abs(spike_sites - b * spike_depths) <= 5

        params["spike_times"] = params["spike_times"][i]
        spike_amplitudes = spike_amplitudes[i]
        spike_depths = spike_depths[i]
        spike_sites = spike_sites[i]

    return params["spike_times"], spike_amplitudes, spike_depths, spike_sites


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
    counts = np.bincount(spike_templates, minlength=num_indices)
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


plot_ks_drift_map(
    "/Users/joeziminski/data/bombcelll/sorter_output",
    localised_spikes_only=True,
    weight_histogram_by_amplitude=False,
    only_include_large_amplitude_spikes=True,
    add_histogram_peaks_and_boundaries=True,
    decimate=False,
    add_histogram_plot=True,
)
