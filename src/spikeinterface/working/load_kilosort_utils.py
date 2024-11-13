from __future__ import annotations

from pathlib import Path
from spikeinterface.core import read_python
import numpy as np
import pandas as pd

from scipy import stats

# TODO: spike_times -> spike_indexes


def compute_spike_amplitude_and_depth(
    sorter_output: str | Path,
    localised_spikes_only,
    exclude_noise,
    gain: float | None = None,
    localised_spikes_channel_cutoff: int = None,  # TODO
) -> tuple[np.ndarray, ...]:
    """
    Compute the amplitude and depth of all detected spikes from the kilosort output.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    sorter_output : str | Path
        Path to the kilosort run sorting output.
    localised_spikes_only : bool
        If `True`, only spikes with small spatial footprint (i.e. 20 channels within 1/2 of the
        amplitude of the maximum loading channel) and which are close to the average depth for
        the cluster are returned.
    gain: float | None
        If a float provided, the `spike_amplitudes` will be scaled by this gain.
    localised_spikes_channel_cutoff : int
        If `localised_spikes_only` is `True`, spikes that have less than half of the
        maximum loading channel over a range of n channels are removed.
        This sets the number of channels.

    Returns
    -------
    spike_indexes : np.ndarray
        (num_spikes,) array of spike indexes.
    spike_amplitudes : np.ndarray
        (num_spikes,) array of corresponding spike amplitudes.
    spike_depths : np.ndarray
        (num_spikes,) array of corresponding depths (probe y-axis location).

    Notes
    -----
    In `_template_positions_amplitudes` spike depths is calculated as simply the template
    depth, for each spike (so it is the same for all spikes in a cluster). Here we need
    to find the depth of each individual spike, using its low-dimensional projection.
    `pc_features` (num_spikes, num_PC, num_channels) holds the PC values for each spike.
    Taking the first component, the subset of 32 channels associated with this
    spike  are indexed to get the actual channel locations (in um). Then, the channel
    locations are weighted by their PC values.
    """
    if isinstance(sorter_output, str):
        sorter_output = Path(sorter_output)

    params = _load_ks_dir(sorter_output, load_pcs=True, exclude_noise=exclude_noise)

    if localised_spikes_only:
        localised_templates = []

        for idx, template in enumerate(params["templates"]):
            max_channel = np.max(np.abs(params["templates"][idx, :, :]))
            channels_over_threshold = np.max(np.abs(params["templates"][idx, :, :]), axis=0) > 0.5 * max_channel
            channel_ids_over_threshold = np.where(channels_over_threshold)[0]

            if np.ptp(channel_ids_over_threshold) <= localised_spikes_channel_cutoff:
                localised_templates.append(idx)

        localised_template_by_spike = np.isin(params["spike_templates"], localised_templates)

        params["spike_templates"] = params["spike_templates"][localised_template_by_spike]
        params["spike_indexes"] = params["spike_indexes"][localised_template_by_spike]
        params["spike_clusters"] = params["spike_clusters"][localised_template_by_spike]
        params["temp_scaling_amplitudes"] = params["temp_scaling_amplitudes"][localised_template_by_spike]
        params["pc_features"] = params["pc_features"][localised_template_by_spike]

    # Compute spike depths
    pc_features = params["pc_features"][:, 0, :]
    pc_features[pc_features < 0] = 0

    # Get the channel indexes corresponding to the 32 channels from the PC.
    spike_features_indices = params["pc_features_indices"][params["spike_templates"], :]

    ycoords = params["channel_positions"][:, 1]
    spike_feature_ycoords = ycoords[spike_features_indices]

    spike_depths = np.sum(spike_feature_ycoords * pc_features**2, axis=1) / np.sum(pc_features**2, axis=1)

    spike_feature_coords = params["channel_positions"][spike_features_indices, :]
    norm_weights = pc_features / np.sum(pc_features, axis=1)[:, np.newaxis]  # TOOD: see why they use square
    weighted_locs = spike_feature_coords * norm_weights[:, :, np.newaxis]
    weighted_locs = np.sum(weighted_locs, axis=1)
    #    Amplitude is calculated for each spike as the template amplitude
    #    multiplied by the `template_scaling_amplitudes`.

    # Compute amplitudes, scale if required and drop un-localised spikes before returning.
    spike_amplitudes, _, _, _, unwhite_templates, *_ = _template_positions_amplitudes(
        params["templates"],
        params["whitening_matrix_inv"],
        ycoords,
        params["spike_templates"],
        params["temp_scaling_amplitudes"],
    )

    if gain is not None:
        spike_amplitudes *= gain

    max_site = np.argmax(np.max(np.abs(unwhite_templates), axis=1), axis=1)
    spike_sites = max_site[params["spike_templates"]]

    if localised_spikes_only:
        # Interpolate the channel ids to location.
        # Remove spikes > 5 um from average position
        # Above we already removed non-localized templates, but that on its own is insufficient.
        # Note for IMEC probe adding a constant term kills the regression making the regressors rank deficient
        # TODO: a couple of approaches. 1) do everything in 3D, draw a sphere around prediction, take spikes only within the sphere
        # 2) do separate for x, y. But resolution will be much lower, making things noisier, also harder to determine threshold.
        # 3) just use depth. Probably go for that. check with others.
        spike_depths = weighted_locs[:, 1]
        b = stats.linregress(spike_depths, spike_sites).slope
        i = np.abs(spike_sites - b * spike_depths) <= 5  # TODO: need to expose this

        params["spike_indexes"] = params["spike_indexes"][i]
        spike_amplitudes = spike_amplitudes[i]
        weighted_locs = weighted_locs[i, :]

    return params["spike_indexes"], spike_amplitudes, weighted_locs, spike_sites  # TODO: rename everything


def _filter_large_amplitude_spikes(
    spike_times: np.ndarray,
    spike_amplitudes: np.ndarray,
    spike_depths: np.ndarray,
    large_amplitude_only_segment_size,
) -> tuple[np.ndarray, ...]:
    """
    Return spike properties with only the largest-amplitude spikes included. The probe
    is split into egments, and within each segment the mean and std computed.
    Any spike less than 1.5x the standard deviation in amplitude of it's segment is excluded
    Splitting the probe is only done for the exclusion step, the returned array are flat.

    Takes as input arrays `spike_times`, `spike_depths` and `spike_amplitudes` and returns
    copies of these arrays containing only the large amplitude spikes.
    """
    spike_bool = np.zeros_like(spike_amplitudes, dtype=bool)

    segment_size_um = large_amplitude_only_segment_size
    probe_segments_left_edges = np.arange(np.floor(spike_depths.max() / segment_size_um) + 1) * segment_size_um

    for segment_left_edge in probe_segments_left_edges:
        segment_right_edge = segment_left_edge + segment_size_um

        spikes_in_seg = np.where(np.logical_and(spike_depths >= segment_left_edge, spike_depths < segment_right_edge))[
            0
        ]
        spike_amps_in_seg = spike_amplitudes[spikes_in_seg]
        is_high_amplitude = spike_amps_in_seg > np.mean(spike_amps_in_seg) + 1.5 * np.std(spike_amps_in_seg, ddof=1)

        spike_bool[spikes_in_seg] = is_high_amplitude

    spike_times = spike_times[spike_bool]
    spike_amplitudes = spike_amplitudes[spike_bool]
    spike_depths = spike_depths[spike_bool]

    return spike_times, spike_amplitudes, spike_depths


def _template_positions_amplitudes(
    templates: np.ndarray,
    inverse_whitening_matrix: np.ndarray,
    ycoords: np.ndarray,
    spike_templates: np.ndarray,
    template_scaling_amplitudes: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Calculate the amplitude and depths of (unwhitened) templates and spikes.
    Amplitude is calculated for each spike as the template amplitude
    multiplied by the `template_scaling_amplitudes`.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    templates : np.ndarray
        (num_clusters, num_samples, num_channels) array of templates.
    inverse_whitening_matrix: np.ndarray
        Inverse of the whitening matrix used in KS preprocessing, used to
        unwhiten templates.
    ycoords : np.ndarray
        (num_channels,) array of the y-axis (depth) channel positions.
    spike_templates : np.ndarray
        (num_spikes,) array indicating the template associated with each spike.
    template_scaling_amplitudes : np.ndarray
        (num_spikes,) array holding the scaling amplitudes, by which the
        template was scaled to match each spike.

    Returns
    -------
    spike_amplitudes : np.ndarray
        (num_spikes,) array of the amplitude of each spike.
    spike_depths : np.ndarray
        (num_spikes,) array of the depth (probe y-axis) of each spike. Note
        this is just the template depth for each spike (i.e. depth of all spikes
        from the same cluster are identical).
    template_amplitudes : np.ndarray
        (num_templates,) Amplitude of each template, calculated as average of spike amplitudes.
    template_depths : np.ndarray
        (num_templates,) array of the depth of each template.
    unwhite_templates : np.ndarray
        Unwhitened templates (num_clusters, num_samples, num_channels).
    trough_peak_durations : np.ndarray
        (num_templates, ) array of durations from trough to peak for each template waveform
    waveforms : np.ndarray
        (num_templates, num_samples) Waveform of each template, taken as the signal on the maximum loading channel.
    """
    # Unwhiten the template waveforms
    unwhite_templates = np.zeros_like(templates)
    for idx, template in enumerate(templates):
        unwhite_templates[idx, :, :] = templates[idx, :, :] @ inverse_whitening_matrix

    # First, calculate the depth of each template from the amplitude
    # on each channel by the center of mass method.

    # Take the max amplitude for each channel, then use the channel
    # with most signal as template amplitude. Zero any small channel amplitudes.
    template_amplitudes_per_channel = np.max(unwhite_templates, axis=1) - np.min(unwhite_templates, axis=1)

    template_amplitudes_unscaled = np.max(template_amplitudes_per_channel, axis=1)

    threshold_values = 0.3 * template_amplitudes_unscaled
    template_amplitudes_per_channel[template_amplitudes_per_channel < threshold_values[:, np.newaxis]] = 0

    # Calculate the template depth as the center of mass based on channel amplitudes
    template_depths = np.sum(template_amplitudes_per_channel * ycoords[np.newaxis, :], axis=1) / np.sum(
        template_amplitudes_per_channel, axis=1
    )

    # Next, find the depth of each spike based on its template. Recompute the template
    # amplitudes as the average of the spike amplitudes ('since
    # tempScalingAmps are equal mean for all templates')
    spike_amplitudes = template_amplitudes_unscaled[spike_templates] * template_scaling_amplitudes

    # Take the average of all spike amplitudes to get actual template amplitudes
    # (since tempScalingAmps are equal mean for all templates)
    num_indices = templates.shape[0]
    sum_per_index = np.zeros(num_indices, dtype=np.float64)
    np.add.at(sum_per_index, spike_templates, spike_amplitudes)
    counts = np.bincount(spike_templates, minlength=num_indices)
    template_amplitudes = np.divide(sum_per_index, counts, out=np.zeros_like(sum_per_index), where=counts != 0)

    # Each spike's depth is the depth of its template
    spike_depths = template_depths[spike_templates]

    # Get channel with the largest amplitude (take that as the waveform)
    max_site = np.argmax(np.max(np.abs(templates), axis=1), axis=1)

    # Use template channel with max signal as waveform
    waveforms = np.empty(templates.shape[:2])
    for idx, template in enumerate(templates):
        waveforms[idx, :] = templates[idx, :, max_site[idx]]

    # Get trough-to-peak time for each template. Find the trough as the
    # minimum signal for the template waveform. The duration (in
    # samples) is the num samples from trough to the largest value
    # following the trough.
    waveform_trough = np.argmin(waveforms, axis=1)

    trough_peak_durations = np.zeros(waveforms.shape[0])
    for idx, tmp_max in enumerate(waveforms):
        trough_peak_durations[idx] = np.argmax(tmp_max[waveform_trough[idx] :])

    return (
        spike_amplitudes,
        spike_depths,
        template_depths,
        template_amplitudes,
        unwhite_templates,
        trough_peak_durations,
        waveforms,
    )


def _load_ks_dir(sorter_output: Path, exclude_noise: bool = True, load_pcs: bool = False) -> dict:
    """
    Loads the output of Kilosort into a `params` dict.

    This function was ported from Nick Steinmetz's `spikes` repository MATLAB
    code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    sorter_output : Path
        Path to the kilosort run sorting output.
    exclude_noise : bool
        If `True`, units labelled as "noise` are removed from all
        returned arrays (i.e. both units and associated spikes are dropped).
    load_pcs : bool
        If `True`, principal component (PC) features are loaded.

    Parameters
    ----------
    params : dict
        A dictionary of parameters combining both the kilosort `params.py`
        file as data loaded from `npy` files. The contents of the `npy`
        files can be found in the Phy documentation.

    Notes
    -----
    When merging and splitting in `Phy`, all changes are made to the
    `spike_clusters.npy` (cluster assignment per spike) and `cluster_groups`
    csv/tsv which contains the quality assignment (e.g. "noise") for each cluster.
    As this function strips the spikes and units based on only these two
    data structures, they will work following manual reassignment in Phy.
    """
    sorter_output = Path(sorter_output)

    params = read_python(sorter_output / "params.py")

    spike_indexes = np.load(sorter_output / "spike_times.npy")
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
    if exclude_noise and (
        (cluster_path := sorter_output / "cluster_groups.csv").is_file()
        or (cluster_path := sorter_output / "cluster_group.tsv").is_file()
    ):
        cluster_ids, cluster_groups = _load_cluster_groups(cluster_path)

        noise_cluster_ids = cluster_ids[cluster_groups == 0]
        not_noise_clusters_by_spike = ~np.isin(spike_clusters.ravel(), noise_cluster_ids)

        spike_indexes = spike_indexes[not_noise_clusters_by_spike]
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
        "spike_indexes": spike_indexes.squeeze(),
        "spike_templates": spike_templates.squeeze(),
        "spike_clusters": spike_clusters.squeeze(),
        "pc_features": pc_features,
        "pc_features_indices": pc_features_indices,
        "temp_scaling_amplitudes": temp_scaling_amplitudes.squeeze(),
        "cluster_ids": cluster_ids,
        "cluster_groups": cluster_groups,
        "channel_positions": np.load(sorter_output / "channel_positions.npy"),
        "templates": np.load(sorter_output / "templates.npy"),
        "whitening_matrix_inv": np.load(sorter_output / "whitening_mat_inv.npy"),
    }
    params.update(new_params)

    return params


def _load_cluster_groups(cluster_path: Path) -> tuple[np.ndarray, ...]:
    """
    Load kilosort `cluster_groups` file, that contains a table of
    quality assignments, one per unit. These can be "noise", "mua", "good"
    or "unsorted".

    There is some slight formatting differences between the `.tsv` and `.csv`
    versions, presumably from different kilosort versions.

    This function was ported from Nick Steinmetz's `spikes` repository MATLAB code,
    https://github.com/cortex-lab/spikes

    Parameters
    ----------
    cluster_path : Path
        The full filepath to the `cluster_groups` tsv or csv file.

    Returns
    -------
    cluster_ids : np.ndarray
        (num_clusters,) Array of (integer) unit IDs.

    cluster_groups : np.ndarray
        (num_clusters,) Array of (integer) unit quality assignments, see code
        below for mapping to "noise", "mua", "good" and "unsorted".
    """
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
