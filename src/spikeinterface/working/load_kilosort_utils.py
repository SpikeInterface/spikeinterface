from __future__ import annotations

from pathlib import Path
from spikeinterface.core import read_python
import numpy as np
import pandas as pd

from scipy import stats

"""
Notes
-----
- not everything is used for current purposes
- things might be useful in future for making a sorting analyzer - compute template amplitude as average of spike amplitude.

TODO: testing against diferent ks versions
"""

########################################################################################################################
# Get Spike Data
########################################################################################################################


def compute_spike_amplitude_and_depth(
    params: dict,
    localised_spikes_only,
    gain: float | None = None,
    localised_spikes_channel_cutoff: int = None,
) -> tuple[np.ndarray, ...]:
    """
    Compute the indicies, amplitudes and locations for all detected spikes from the kilosort output.

    This function is based on code in Nick Steinmetz's `spikes` repository,
    https://github.com/cortex-lab/spikes

    Parameters
    ----------
    params : dict
        `params` as loaded from the kilosort output directory (see `load_ks_dir()`)
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
    spike_indices : np.ndarray
        (num_spikes,) array of spike indices.
    spike_amplitudes : np.ndarray
        (num_spikes,) array of corresponding spike amplitudes.
    spike_locations : np.ndarray
        (num_spikes, 2) array of corresponding spike locations (x, y) estimated using
        center of mass from the first PC (or, second PC if no signal on first PC).
        See `_get_locations_from_pc_features()` for details.
    """
    if isinstance(sorter_output, str):
        sorter_output = Path(sorter_output)

    if not params["pc_features"]:
        raise ValueError("`pc_features` must be loaded into params. Use `load_ks_dir` with `load_pcs=True`.")

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
        params["spike_indices"] = params["spike_indices"][localised_template_by_spike]
        params["spike_clusters"] = params["spike_clusters"][localised_template_by_spike]
        params["temp_scaling_amplitudes"] = params["temp_scaling_amplitudes"][localised_template_by_spike]
        params["pc_features"] = params["pc_features"][localised_template_by_spike]

    # Compute the spike locations and maximum-loading channel per spike
    spike_locations, spike_max_sites = _get_locations_from_pc_features(params)

    # Amplitude is calculated for each spike as the template amplitude
    # multiplied by the `template_scaling_amplitudes`.
    template_amplitudes_unscaled, *_ = get_unwhite_template_info(
        params["templates"],
        params["whitening_matrix_inv"],
        params["channel_positions"],
    )
    spike_amplitudes = template_amplitudes_unscaled[params["spike_templates"]] * params["temp_scaling_amplitudes"]

    if gain is not None:
        spike_amplitudes *= gain

    if localised_spikes_only:
        # Interpolate the channel ids to location.
        # Remove spikes > 5 um from average position
        # Above we already removed non-localized templates, but that on its own is insufficient.
        # Note for IMEC probe adding a constant term kills the regression making the regressors rank deficient
        spike_depths = spike_locations[:, 1]
        b = stats.linregress(spike_depths, spike_max_sites).slope
        i = np.abs(spike_max_sites - b * spike_depths) <= 5

        params["spike_indices"] = params["spike_indices"][i]
        spike_amplitudes = spike_amplitudes[i]
        spike_locations = spike_locations[i, :]
        spike_max_sites = spike_max_sites[i]

    return params["spike_indices"], spike_amplitudes, spike_locations, spike_max_sites


def _get_locations_from_pc_features(params):
    """

    Notes
    -----
    Location of of each individual spike is computed from its low-dimensional projection.
    `pc_features` (num_spikes, num_PC, num_channels) holds the PC values for each spike.
    Taking the first component, the subset of 32 channels associated with this
    spike  are indexed to get the actual channel locations (in um). Then, the channel
    locations are weighted by their PC values.

    This function is based on code in Nick Steinmetz's `spikes` repository,
    https://github.com/cortex-lab/spikes
    """
    # Compute spike depths
    pc_features = params["pc_features"][:, 0, :]
    pc_features[pc_features < 0] = 0

    # Some spikes do not load at all onto the first PC. To avoid biasing the
    # dataset by removing these, we repeat the above for the next PC,
    # to compute distances for neurons that do not load onto the 1st PC.
    # This is not ideal at all, it would be much better to a) find the
    # max value for each channel on each of the PCs (i.e. basis vectors).
    # Then recompute the estimated waveform peak on each channel by
    # summing the PCs by their respective weights. However, the PC basis
    # vectors themselves do not appear to be output by KS.

    # We include the (n_channels i.e. features) from the second PC
    # into the `pc_features` mostly containing the first PC. As all
    # operations are per-spike (i.e. row-wise)
    no_pc1_signal_spikes = np.where(np.sum(pc_features, axis=1) == 0)

    pc_features_2 = params["pc_features"][:, 1, :]
    pc_features_2[pc_features_2 < 0] = 0

    pc_features[no_pc1_signal_spikes] = pc_features_2[no_pc1_signal_spikes]

    if any(np.sum(pc_features, axis=1) == 0):
        raise RuntimeError(
            "Some spikes do not load at all onto the first"
            "or second principal component. It is necessary"
            "to extend this code section to handle more components."
        )

    # Get the channel indices corresponding to the 32 channels from the PC.
    spike_features_indices = params["pc_features_indices"][params["spike_templates"], :]

    # Compute the spike locations as the center of mass of the PC scores
    spike_feature_coords = params["channel_positions"][spike_features_indices, :]
    norm_weights = pc_features / np.sum(pc_features, axis=1)[:, np.newaxis]

    spike_locations = spike_feature_coords * norm_weights[:, :, np.newaxis]
    spike_locations = np.sum(spike_locations, axis=1)

    # Find the max site as the channel with the largest PC weight.
    spike_max_sites = spike_features_indices[
        np.arange(spike_features_indices.shape[0]), np.argmax(norm_weights, axis=1)
    ]

    return spike_locations, spike_max_sites


########################################################################################################################
# Get Template Data
########################################################################################################################


def get_unwhite_template_info(
    templates: np.ndarray,
    inverse_whitening_matrix: np.ndarray,
    channel_positions: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Calculate the amplitude and depths of (unwhitened) templates and spikes.
    Amplitude is calculated for each spike as the template amplitude
    multiplied by the `template_scaling_amplitudes`.

    This function is based on code in Nick Steinmetz's `spikes` repository,
    https://github.com/cortex-lab/spikes

    Parameters
    ----------
    templates : np.ndarray
        (num_clusters, num_samples, num_channels) array of templates.
    inverse_whitening_matrix: np.ndarray
        Inverse of the whitening matrix used in KS preprocessing, used to
        unwhiten templates.
    channel_positions : np.ndarray
        (num_channels, 2) array of the x, y channel positions.

    Returns
    -------
    template_amplitudes_unscaled : np.ndarray
        (num_templates,) array of the unscaled tempalte amplitudes. These can be
        used to calculate spike amplitude with `template_amplitude_scalings`.
    template_locations : np.ndarray
        (num_templates, 2) array of the x, y positions (center of mass) of each template.
    unwhite_templates : np.ndarray
        Unwhitened templates (num_clusters, num_samples, num_channels).
    template_max_site : np.array
        The maximum loading spike for the unwhitened template.
    trough_peak_durations : np.ndarray
        (num_templates, ) array of durations from trough to peak for each template waveform
    waveforms : np.ndarray
        (num_templates, num_samples) Waveform of each template, taken as the signal on the maximum loading channel.
    """
    # Unwhiten the template waveforms
    unwhite_templates = np.zeros_like(templates)
    for idx, template in enumerate(templates):
        unwhite_templates[idx, :, :] = templates[idx, :, :] @ inverse_whitening_matrix

    # Take the max amplitude for each channel, then use the channel
    # with most signal as template amplitude.
    template_amplitudes_per_channel = np.max(unwhite_templates, axis=1) - np.min(unwhite_templates, axis=1)

    template_amplitudes_unscaled = np.max(template_amplitudes_per_channel, axis=1)

    # Calculate the template depth as the center of mass based on channel amplitudes
    weights = template_amplitudes_per_channel / np.sum(template_amplitudes_per_channel, axis=1)[:, np.newaxis]
    template_locations = weights @ channel_positions

    # Get channel with the largest amplitude (take that as the waveform)
    template_max_site = np.argmax(np.max(np.abs(unwhite_templates), axis=1), axis=1)

    # Use template channel with max signal as waveform
    waveforms = np.empty(unwhite_templates.shape[:2])

    for idx, template in enumerate(unwhite_templates):
        waveforms[idx, :] = unwhite_templates[idx, :, template_max_site[idx]]

    # Get trough-to-peak time for each template. Find the trough as the
    # minimum signal for the template waveform. The duration (in
    # samples) is the num samples from trough to the largest value
    # following the trough.
    waveform_trough = np.argmin(waveforms, axis=1)

    trough_peak_durations = np.zeros(waveforms.shape[0])
    for idx, tmp_max in enumerate(waveforms):
        trough_peak_durations[idx] = np.argmax(tmp_max[waveform_trough[idx] :])

    return (
        template_amplitudes_unscaled,
        template_locations,
        template_max_site,
        unwhite_templates,
        trough_peak_durations,
        waveforms,
    )


def compute_template_amplitudes_from_spikes(templates, spike_templates, spike_amplitudes):
    """
    Take the average of all spike amplitudes to get actual template amplitudes
    (since tempScalingAmps are equal mean for all templates)

    This function is ported from Nick Steinmetz's `spikes` repository,
    https://github.com/cortex-lab/spikes
    """
    num_indices = templates.shape[0]
    sum_per_index = np.zeros(num_indices, dtype=np.float64)
    np.add.at(sum_per_index, spike_templates, spike_amplitudes)
    counts = np.bincount(spike_templates, minlength=num_indices)
    template_amplitudes = np.divide(sum_per_index, counts, out=np.zeros_like(sum_per_index), where=counts != 0)
    return template_amplitudes


########################################################################################################################
# Load Parameters from KS Directory
########################################################################################################################


def load_ks_dir(sorter_output: Path, exclude_noise: bool = True, load_pcs: bool = False) -> dict:
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

    spike_indices = np.load(sorter_output / "spike_times.npy")
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

        spike_indices = spike_indices[not_noise_clusters_by_spike]
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
        "spike_indices": spike_indices.squeeze(),
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
