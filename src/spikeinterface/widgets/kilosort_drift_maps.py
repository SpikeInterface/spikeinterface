from pathlib import Path

import matplotlib.axis
import scipy.signal
from spikeinterface.core import read_python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats


def plot_ks_drift_map(
    sorter_output: str | Path,
    only_include_large_amplitude_spikes: bool = True,
    decimate: None | int = None,
    add_histogram_plot: bool = False,
    add_histogram_peaks_and_boundaries: bool = True,
    add_drift_events: bool = True,
    weight_histogram_by_amplitude: bool = False,
    localised_spikes_only: bool = False,
    exclude_noise: bool = False,
    gain: float | None = None,
) -> None:
    """
    Create a drift map plot in the kilosort style. This is ported from Nick Steinmetz's
    `spikes` repository MATLAB code, https://github.com/cortex-lab/spikes.

    By default, a raster plot is drawn with the y-axis is spike depth and
    x-axis is time. Optionally, a corresponding 2D activity histogram can be
    added as a subplot (spatial bins, spike counts) with optional
    peak coloring and drift event detection (see below).

    Parameters
    ----------
    sorter_output : str | Path,
        Path to the kilosort output folder.
    only_include_large_amplitude_spikes : bool
        If `True`, only spikes with larger amplitudes are included. For
        details, see `filter_large_amplitude_spikes()`.
    decimate : bool | int
        If an integer n, every nth spike is dropped from the plot. Useful for improving
        performance when there are many spikes. If `None`, spikes will not be decimated.
    add_histogram_plot : bool
        If `True`, an activity histogram will be added to a new subplot to the
        left of the drift map.
    add_histogram_peaks_and_boundaries : bool
        If `True`, activity histogram peaks are detected and colored rec
        if isolated according to start/end boundaries of the peak.
    add_drift_events : bool
        If `True`, drift events will be plot on the raster map. Required
        `add_histogram_plot` and `add_histogram_peaks_and_boundaries` to run.
    weight_histogram_by_amplitude : bool
        If `True`, histogram counts will be weighted by spike amplitude.
    localised_spikes_only : bool
        If `True`, only spatially isolated spikes will be included.
    exclude_noise : bool
        If `True`, units labelled as noise inthe `cluster_groups` file
        will be excluded.
    gain : float | None
        If not `None`, amplitudes will be scaled by the supplied gain.
    """
    spike_times, spike_amplitudes, spike_depths, _ = _compute_spike_amplitude_and_depth(
        sorter_output, localised_spikes_only, exclude_noise, gain
    )

    # Calculate the amplitude range for plotting first, so the scale is always the
    # same across all options (e.g. decimation) which helps with interpretability.
    if only_include_large_amplitude_spikes:
        amplitude_range_all_spikes = (
            spike_amplitudes.min(),
            spike_amplitudes.max(),
        )
    else:
        amplitude_range_all_spikes = np.percentile(spike_amplitudes, (1, 90))

    if decimate:
        spike_times = spike_times[::decimate]
        spike_amplitudes = spike_amplitudes[::decimate]
        spike_depths = spike_depths[::decimate]

    if only_include_large_amplitude_spikes:
        spike_times, spike_depths, spike_amplitudes = filter_large_amplitude_spikes(
            spike_times, spike_depths, spike_amplitudes
        )

    # Setup axis and plot the raster drift map
    fig = plt.figure(figsize=(10, 10 * (6 / 8)))

    if add_histogram_plot:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])
        hist_axis = fig.add_subplot(gs[0])
        raster_axis = fig.add_subplot(gs[1], sharey=hist_axis)
    else:
        raster_axis = fig.add_subplot()

    plot_kilosort_drift_map_raster(
        spike_times,
        spike_amplitudes,
        spike_depths,
        amplitude_range_all_spikes,
        axis=raster_axis,
    )

    if not add_histogram_plot:
        raster_axis.set_xlabel("time")
        raster_axis.set_ylabel("y position")
        plt.show()
        return

    # If the histogram plot is requested, plot it alongside
    # it's peak colouring, bounds display and drift point display.
    hist_axis.set_xlabel("count")
    raster_axis.set_xlabel("time")
    hist_axis.set_ylabel("y position")

    bin_centers, counts = compute_activity_histogram(spike_depths, spike_amplitudes, weight_histogram_by_amplitude)
    hist_axis.plot(counts, bin_centers, color="black", linewidth=1)

    if add_histogram_peaks_and_boundaries:
        drift_events = color_histogram_peaks_and_detect_drift_events(
            spike_times, spike_depths, counts, bin_centers, hist_axis
        )

        if add_drift_events and np.any(drift_events):
            raster_axis.scatter(drift_events[:, 0], drift_events[:, 1], facecolors="r", edgecolors="none")
            for i, _ in enumerate(drift_events):
                raster_axis.text(
                    drift_events[i, 0] + 1, drift_events[i, 1], str(np.round(drift_events[i, 2])), color="r"
                )
    plt.show()


def plot_kilosort_drift_map_raster(
    spike_times: np.ndarray,
    spike_amplitudes: np.ndarray,
    spike_depths: np.ndarray,
    amplitude_range: np.ndarray | tuple,
    axis: matplotlib.axes.Axes,
) -> None:
    """
    Plot a drift raster plot in the kilosort style.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    spike_times : np.ndarray
        (num_spikes,) array of spike times.
    spike_amplitudes : np.ndarray
            (num_spikes,) array of corresponding spike amplitudes.
    spike_depths : np.ndarray
            (num_spikes,) array of corresponding spike depths.
    amplitude_range : np.ndarray
        (2,) array of min, max amplitude values for color binning.
    axis : matplotlib.axes.Axes
        Matplotlib axes object on which to plot the drift map.
    """
    n_color_bins = 20
    marker_size = 0.5

    color_bins = np.linspace(amplitude_range[0], amplitude_range[1], n_color_bins)

    colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]

    for bin_idx in range(n_color_bins - 1):

        spikes_in_amplitude_bin = np.logical_and(
            spike_amplitudes >= color_bins[bin_idx], spike_amplitudes <= color_bins[bin_idx + 1]
        )
        axis.scatter(
            spike_times[spikes_in_amplitude_bin],
            spike_depths[spikes_in_amplitude_bin],
            color=colors[bin_idx],
            s=marker_size,
            antialiased=True,
        )


def compute_activity_histogram(
    spike_depths: np.ndarray, spike_amplitudes: np.ndarray, weight_histogram_by_amplitude: bool
) -> tuple[np.ndarray, ...]:
    """
    Compute the activity histogram for the kilosort drift map left plot.

    Parameters
    ----------
    spike_depths : np.ndarray
        (num_spikes,) array of spike depths.
    spike_amplitudes : np.ndarray
        (num_spikes,) array of spike amplitudes
    weight_histogram_by_amplitude : bool
        If `True`, the spike amplitudes are taken into consideration when generating the
        histogram. The amplitudes are scaled to the range [0, 1] then summed for each bin,
        to generate the histogram values. If `False`, counts (i.e. num spikes per bin)
        are used.

    Returns
    -------
    bin_centers : np.ndarray
        The spatial bin centers (i.e. probe depth) for the histogram.
    values : np.ndarray
        The histogram values. If `weight_histogram_by_amplitude` is `False`, these
        are counts, otherwise they are counts weighted by amplitude.
    """
    assert spike_amplitudes.dtype == np.float64, "`spike amplitudes should be high precision as many values are summed."

    bin_um = 2
    bins = np.arange(spike_depths.min() - bin_um, spike_depths.max() + bin_um, bin_um)
    values, bins = np.histogram(spike_depths, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if weight_histogram_by_amplitude:
        bin_indices = np.digitize(spike_depths, bins, right=True) - 1
        values = np.zeros(bin_indices.max() + 1, dtype=np.float64)
        spike_amplitudes = (spike_amplitudes - spike_amplitudes.min()) / np.ptp(spike_amplitudes)
        np.add.at(values, bin_indices, spike_amplitudes)

    return bin_centers, values


def color_histogram_peaks_and_detect_drift_events(
    spike_times: np.ndarray,
    spike_depths: np.ndarray,
    counts: np.ndarray,
    bin_centers: np.ndarray,
    hist_axis: matplotlib.axes.Axes,
) -> np.ndarray:
    """
    Given an activity histogram, color the peaks red (isolated peak) or
    blue (peak overlaps with other peaks) and compute spatial drift
    events for isolated peaks across time bins.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    spike_times : np.ndarray
        (num_spikes,) array of spike times.
    spike_depths : np.ndarray
        (num_spikes,) array of corresponding spike depths.
    counts : np.ndarray
        (num_bins,) array of histogram bin counts.
    bin_centers : np.ndarray
        (num_bins,) array of histogram bin centers.
    hist_axis : matplotlib.axes.Axes
        Axes on which the histogram is plot, to add peaks.

    Returns
    -------
    drift_events : np.ndarray
        A (num_drift_events, 3) array of drift events. The columns are
        (time_position, spatial_position, drift_value). The drift
        value is computed per time, spatial bin as the difference between
        the median position of spikes in the bin, and the bin center.
    """
    all_peak_indexes = scipy.signal.find_peaks(
        counts,
    )[0]

    # Filter low-frequency peaks, so they are not included in the
    # step to determine whether peaks are overlapping (new step
    # introduced in the port to python)
    bin_above_freq_threshold = counts[all_peak_indexes] > 0.3 * spike_times[-1]
    filtered_peak_indexes = all_peak_indexes[bin_above_freq_threshold]

    drift_events = []
    for idx, peak_index in enumerate(filtered_peak_indexes):

        peak_count = counts[peak_index]

        # Find the start and end of peak min/max bounds (5% of amplitude)
        start_position = np.where(counts[:peak_index] < peak_count * 0.05)[0].max()
        end_position = np.where(counts[peak_index:] < peak_count * 0.05)[0].min() + peak_index

        if (  # bounds include another, different histogram peak
            idx > 0
            and start_position < filtered_peak_indexes[idx - 1]
            or idx < filtered_peak_indexes.size - 1
            and end_position > filtered_peak_indexes[idx + 1]
        ):
            hist_axis.scatter(peak_count, bin_centers[peak_index], facecolors="none", edgecolors="blue")
            continue

        else:
            for position in [start_position, end_position]:
                hist_axis.axhline(bin_centers[position], 0, counts.max(), color="grey", linestyle="--")
            hist_axis.scatter(peak_count, bin_centers[peak_index], facecolors="none", edgecolors="red")

            # For isolated histogram peaks, detect the drift events, defined as
            # difference between spatial bin center and median spike depth in the bin
            # over 6 um (in time / spatial bins with at least 10 spikes).
            depth_in_window = np.logical_and(
                spike_depths > bin_centers[start_position],
                spike_depths < bin_centers[end_position],
            )
            current_spike_depths = spike_depths[depth_in_window]
            current_spike_times = spike_times[depth_in_window]

            window_s = 10

            all_time_bins = np.arange(0, np.ceil(spike_times[-1]).astype(int), window_s)
            for time_bin in all_time_bins:

                spike_in_time_bin = np.logical_and(
                    current_spike_times >= time_bin, current_spike_times <= time_bin + window_s
                )
                drift_size = bin_centers[peak_index] - np.median(current_spike_depths[spike_in_time_bin])

                # 6 um is the hardcoded threshold for drift, and we want at least 10 spikes for the median calculation
                bin_has_drift = np.abs(drift_size) > 6 and np.sum(spike_in_time_bin, dtype=np.int16) > 10
                if bin_has_drift:
                    drift_events.append((time_bin + window_s / 2, bin_centers[peak_index], drift_size))

    drift_events = np.array(drift_events)

    return drift_events


def _compute_spike_amplitude_and_depth(
    sorter_output: str | Path, localised_spikes_only, exclude_noise, gain: float | None = None
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
        the unit are returned.
    gain: float | None
        If a float provided, the `spike_amplitudes` will be scaled by this gain.

    Returns
    -------
    spike_times : np.ndarray
        (num_spikes,) array of spike times.
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

    params = load_ks_dir(sorter_output, load_pcs=True, exclude_noise=exclude_noise)

    if localised_spikes_only:
        localised_templates = []

        for idx, template in enumerate(params["templates"]):
            max_channel = np.max(np.abs(params["templates"][idx, :, :]))
            channels_over_threshold = np.max(np.abs(params["templates"][idx, :, :]), axis=0) > 0.5 * max_channel
            channel_ids_over_threshold = np.where(channels_over_threshold)[0]

            if np.ptp(channel_ids_over_threshold) <= 20:
                localised_templates.append(idx)

        localised_template_by_spike = np.isin(params["spike_templates"], localised_templates)

        params["spike_templates"] = params["spike_templates"][localised_template_by_spike]
        params["spike_times"] = params["spike_times"][localised_template_by_spike]
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
        b = stats.linregress(spike_depths, spike_sites).slope
        i = np.abs(spike_sites - b * spike_depths) <= 5

        params["spike_times"] = params["spike_times"][i]
        spike_amplitudes = spike_amplitudes[i]
        spike_depths = spike_depths[i]

    return params["spike_times"], spike_amplitudes, spike_depths, spike_sites


def filter_large_amplitude_spikes(
    spike_times: np.ndarray, spike_depths: np.ndarray, spike_amplitudes: np.ndarray
) -> tuple[np.ndarray, ...]:
    """
    Return spike properties with only the largest-amplitude spikes included. The probe
    is split into 800 um segments, and within each segment the mean and std computed.
    Any spike less than 1.5x the standard deviation in amplitude of it's segment is excluded
    Splitting the probe is only done for the exclusion step, the returned array are flat.

    Takes as input arrays `spike_times`, `spike_depths` and `spike_amplitudes` and returns
    copies of these arrays containing only the large amplitude spikes.
    """
    spike_bool = np.zeros_like(spike_amplitudes, dtype=bool)

    segment_size_um = 800
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
    spike_depths = spike_depths[spike_bool]
    spike_amplitudes = spike_amplitudes[spike_bool]

    return spike_times, spike_depths, spike_amplitudes


def _template_positions_amplitudes(
    templates: np.ndarray,
    inverse_whitening_matrix: np.ndarray,
    ycoords: np.ndarray,
    spike_templates: np.ndarray,
    template_scaling_amplitudes: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Calculate the amplitude and depths of (unwhitened) templates and spikes.

    This function was ported from Nick Steinmetz's `spikes` repository
    MATLAB code, https://github.com/cortex-lab/spikes

    Parameters
    ----------
    templates : np.ndarray
        (num_clusters x num_samples x num_channels) array of templates.
    inverse_whitening_matrix: np.ndarray
        Inverse of the whitening matrix used in KS preprocessing, used to
        unwhiten templates.
    ycoords : np.ndarray
        (num_channels) array of the y-axis (depth) channel positions.
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
        (num_templates, ) Duration from trough to peak for the template waveform
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
        files can be found on the Phy documentation.

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
    if exclude_noise and (
        (cluster_path := sorter_output / "cluster_groups.csv").is_file()
        or (cluster_path := sorter_output / "cluster_group.tsv").is_file()
    ):
        cluster_ids, cluster_groups = _load_cluster_groups(cluster_path)

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
        "spike_times": spike_times.squeeze(),
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


plot_ks_drift_map(
    "/Users/joeziminski/data/bombcelll/sorter_output",
    localised_spikes_only=False,
    weight_histogram_by_amplitude=False,
    only_include_large_amplitude_spikes=True,
    add_histogram_peaks_and_boundaries=True,
    decimate=False,
    add_histogram_plot=True,
    exclude_noise=True,
)
