from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import periodogram

from spikeinterface.core import BaseRecording, load
from spikeinterface.core.job_tools import fix_job_kwargs, TimeSeriesChunkExecutor

from spikeinterface.preprocessing import common_reference, highpass_filter
import spikeinterface.full as si

si.set_global_job_kwargs(n_jobs=8)

SAVE_PLOTS = False  # False opens the figures instead of saving them.
PLOT_FOLDER = Path(__file__).resolve().parent / "plots" / "raw_data_quality"


def _compute_rolling_rmse(data, num_chunks_for_baseline):
    """Return each chunk's RMSE from the mean of the first baseline chunks.

    Average squared differences over all non-time axes (channels, bins, etc.).
    """
    num_chunks = data.shape[0]
    baseline = np.mean(data[:num_chunks_for_baseline, :], axis=0)

    rolling_change_rmse = np.empty(num_chunks)
    for chunk_idx in range(num_chunks):
        rolling_change_rmse[chunk_idx] = np.sqrt(np.mean((data[chunk_idx] - baseline)**2))

    return rolling_change_rmse


def _compute_rolling_rmse_2d(data, num_chunks_for_baseline, sigma=(0.5, 0.5), log_base=None):
    """Compare chunk maps after optional log(1 + data) and Gaussian smoothing.

    data has shape (chunks, map_axis_0, map_axis_1). Sigma is in bins
    along the two map axes; chunks are never smoothed together. log_base
    may be None, 2, or 10. The baseline is computed after transformation.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("Expected data with shape (chunks, map_axis_0, map_axis_1)")
    if log_base is not None:
        if log_base not in (2, 10):
            raise ValueError("log_base must be None, 2, or 10")
        if np.any(data < 0):
            raise ValueError("Log transformation requires nonnegative data")
        data = np.log1p(data) / np.log(log_base)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (2,))
    data = gaussian_filter(data, sigma=(0.0, *sigma), mode="nearest")
    return _compute_rolling_rmse(data, num_chunks_for_baseline)


def _compute_channel_xcorr(
    recording, num_chunks=100, chunk_size_s=1.0, highpass_hz=300,
    correlation_threshold=0.3, distance_tolerance_fraction=0.05, seed=0, reference=None,
):
    """Sample zero-lag Pearson channel correlations with optional median referencing.

    Draw num_chunks random chunks per segment using get_random_data_chunks.
    Set reference="median" for CMR after high-pass filtering. Identical sampling
    settings and seed select identical chunks before and after referencing.
    Percentages describe valid sampled chunk/pair observations, not continuous
    event durations. Constant channels yield NaN, and diagonals are excluded from
    summaries. Distance bands use y-depth separations within each channel group,
    relative to that group's recorded depth span (not the physical probe length).
    """
    # TODO: Think more about chunk duration, frequency band, lagged/absolute
    # correlations, threshold calibration, distance bands and shank grouping.
    from spikeinterface.core import get_random_data_chunks

    # Filter and sample: retain high frequencies and draw separate random chunks.
    num_channels = recording.get_num_channels()
    filtered_recording = highpass_filter(recording, freq_min=highpass_hz)
    if reference == "median":
        filtered_recording = common_reference(filtered_recording, operator="median")
    elif reference is not None:
        raise ValueError('reference must be None or "median"')
    chunks = get_random_data_chunks(
        filtered_recording, num_chunks_per_segment=num_chunks,
        chunk_duration=chunk_size_s, seed=seed,
        return_in_uV=True, concatenated=False,
    )
    # Chunk matrices: correlate every channel pair at zero time lag.
    num_chunks = len(chunks)
    matrices = np.empty((num_chunks, num_channels, num_channels), dtype=np.float64)
    for chunk_index, traces in enumerate(chunks):
        # Flat channels have undefined correlation and remain NaN.
        with np.errstate(divide="ignore", invalid="ignore"):
            matrices[chunk_index] = np.corrcoef(traces, rowvar=False)
    matrices = np.clip(matrices, -1, 1)
    # Valid observations: exclude undefined correlations from summary denominators.
    valid = np.isfinite(matrices)
    valid_counts = valid.sum(axis=0)

    # Mean matrix: average each channel pair's correlation across valid chunks.
    mean_matrix = np.divide(
        np.nansum(matrices, axis=0), valid_counts,
        out=np.full((num_channels, num_channels), np.nan), where=valid_counts > 0,
    )

    # Std matrix: measure how much each pair's correlation varies across chunks.
    std_matrix = np.sqrt(np.divide(
        np.nansum((matrices - mean_matrix)**2, axis=0), valid_counts,
        out=np.full_like(mean_matrix, np.nan), where=valid_counts > 0,
    ))

    # High-correlation percentage: fraction of valid chunks above the positive threshold.
    high = valid & (matrices >= correlation_threshold)

    high_percent_matrix = np.divide(
        100.0 * high.sum(axis=0), valid_counts,
        out=np.full_like(mean_matrix, np.nan), where=valid_counts > 0,
    )
    # Per-channel summaries: average over other channels, excluding self-correlation.
    off_diagonal = ~np.eye(num_channels, dtype=bool)
    partner_counts = np.isfinite(mean_matrix) & off_diagonal
    num_partners = partner_counts.sum(axis=1)
    per_channel = pd.DataFrame(index=pd.Index(recording.channel_ids, name="channel_id"))

    for name, values in (
        ("xcorr_mean_with_other_channels", mean_matrix),
        ("xcorr_mean_temporal_std_with_other_channels", std_matrix),
        ("xcorr_mean_high_percent_with_other_channels", high_percent_matrix),
    ):
        per_channel[name] = np.divide(
            np.where(partner_counts, values, 0).sum(axis=1), num_partners,
            out=np.full(num_channels, np.nan), where=num_partners > 0,
        )

    # Pair distances: express depth separation as a fraction of each group's depth span.
    # Pairs in different groups are left undefined and excluded from distance summaries.
    locations = np.asarray(recording.get_channel_locations())
    depths = locations[:, 1]
    groups = recording.get_channel_groups()
    if groups is None:
        groups = np.zeros(num_channels, dtype=int)
    normalized_distance = np.full_like(mean_matrix, np.nan)

    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        group_depths = depths[indices]
        span = np.ptp(group_depths)
        if np.isfinite(group_depths).all() and span > 0:
            normalized_distance[np.ix_(indices, indices)] = (
                np.abs(group_depths[:, None] - group_depths[None, :]) / span
            )
    # Distance summaries: select pairs near 1/4, 1/3, 1/2 and 2/3 of the depth span.
    # Count each pair once; report high-correlation percentages per chunk and pooled
    # over all valid pair/chunk observations within each distance band.
    distance_results = {}
    for label, fraction in (("quarter", 1 / 4), ("third", 1 / 3), ("half", 1 / 2), ("two_thirds", 2 / 3)):
        pair_mask = np.triu(
            np.abs(normalized_distance - fraction) <= distance_tolerance_fraction, k=1
        )
        valid_per_chunk = valid[:, pair_mask].sum(axis=1)
        high_per_chunk = high[:, pair_mask].sum(axis=1)
        total_valid = valid_per_chunk.sum()
        distance_results[label] = {
            "pair_mask": pair_mask,
            "num_pairs": int(pair_mask.sum()),
            "valid_pair_chunk_count": int(total_valid),
            "high_pair_chunk_count": int(high_per_chunk.sum()),
            "high_percent": 100.0 * high_per_chunk.sum() / total_valid if total_valid else np.nan,
            "high_pair_percent_per_chunk": np.divide(
                100.0 * high_per_chunk, valid_per_chunk,
                out=np.full(num_chunks, np.nan), where=valid_per_chunk > 0,
            ),
        }
    # Outputs: retain the full matrices, summaries and settings for inspection.
    return {
        "matrices": matrices, "mean_matrix": mean_matrix, "std_matrix": std_matrix,
        "high_percent_matrix": high_percent_matrix, "valid_chunk_counts": valid_counts,
        "channel_ids": recording.channel_ids,
        "chunk_size_s": chunks[0].shape[0] / recording.get_sampling_frequency(),
        "highpass_hz": highpass_hz, "correlation_threshold": correlation_threshold,
        "reference": reference,
        "distance_tolerance_fraction": distance_tolerance_fraction, "seed": seed,
        "distance_results": distance_results, "per_channel_results": per_channel,
    }


def _plot_quality_metrics(
    raw_recording, preprocessed_recording, chunk_times_s, raw_chunk_times_s,
    ap_rms_array, ap_rms_linear_stability, ap_hist_array, histogram_centers,
    psd, freqs, rolling_results, xcorr_results, num_chunks_for_baseline, rolling_ap_rms,
    ap_mean_rms_over_time, ap_mean_rms_linear_stability,
):
    """Create all diagnostic figures in one call; save PNGs or show them.

    RMS, histograms and PSD describe segment 0, matching the metric arrays.
    Xcorr includes the random chunks sampled from all segments.
    """
    figures = {}
    plot_files = {}
    depths = preprocessed_recording.get_channel_locations()[:, 1]
    num_plot_channels = min(10, len(depths))
    # Select distinct channels near equally spaced target depths, top to bottom.
    selected_channels = []
    for target in np.linspace(depths.max(), depths.min(), num_plot_channels):
        candidates = np.argsort(np.abs(depths - target))
        selected_channels.append(next(int(i) for i in candidates if i not in selected_channels))
    selected_ids = preprocessed_recording.channel_ids[selected_channels]
    raw_indices = raw_recording.ids_to_indices(selected_ids)
    rows = (num_plot_channels + 1) // 2
    baseline_label = f"Baseline: mean of first {min(num_chunks_for_baseline, len(chunk_times_s))} chunks"
    chunk_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(chunk_times_s)))

    # RMS and linear fits: retain the existing slope units of uV per sampled chunk.
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.8 * rows), squeeze=False, layout="constrained")
    chunk_indices = np.arange(len(chunk_times_s))
    for ax, channel_index in zip(axes.flat, selected_channels):
        slope = ap_rms_linear_stability["slopes"][channel_index]
        r_squared = ap_rms_linear_stability["r_squared"][channel_index]
        values = ap_rms_array[:, channel_index]
        intercept = values.mean() - slope * chunk_indices.mean()
        ax.plot(chunk_times_s, values, "o-", markersize=3, label="Chunk RMS")
        ax.plot(chunk_times_s, intercept + slope * chunk_indices, "--", color="tab:red", label="Linear fit")
        ax.set_title(
            f"Channel {preprocessed_recording.channel_ids[channel_index]} | depth {depths[channel_index]:g} µm\n"
            f"R² = {r_squared:.3f} | slope = {slope:.4g} µV/chunk", fontsize=10
        )
        ax.set(xlabel="Chunk midpoint (s, segment 0)", ylabel="RMS (µV)")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[num_plot_channels:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Preprocessed AP RMS and linear trends")
    figures["01_rms_linear_fits"] = fig

    # Rolling RMS: one channel-aggregated deviation per chunk.
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(chunk_times_s, rolling_results["rolling_rms_change_rmse"], "o-", markersize=3)
    ax.set(title=f"RMS stability\n{baseline_label}", xlabel="Chunk midpoint (s, segment 0)", ylabel="RMSE (µV)")
    ax.grid(alpha=0.2)
    figures["02_rms_stability"] = fig

    # Untransformed amplitude histograms: overlay every chunk on the selected channels.
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.8 * rows), squeeze=False, layout="constrained")
    for ax, channel_index in zip(axes.flat, selected_channels):
        for chunk_index, color in enumerate(chunk_colors):
            ax.plot(histogram_centers, ap_hist_array[chunk_index, channel_index], color=color, alpha=0.6, lw=0.8)
        ax.set_title(f"Channel {preprocessed_recording.channel_ids[channel_index]} | depth {depths[channel_index]:g} µm")
        ax.set(xlabel="Amplitude (µV)", ylabel="Sample count / bin")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[num_plot_channels:]:
        ax.set_visible(False)
    fig.suptitle("Preprocessed AP histograms — all chunks overlaid; edge bins include overflow")
    norm = plt.Normalize(float(chunk_times_s.min()), float(chunk_times_s.max()))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes.flat)[:num_plot_channels],
                 label="Chunk midpoint (s, segment 0)", shrink=0.7)
    figures["03_amplitude_histograms"] = fig

    # Histogram stability: partition counts and full-map errors have separate panels.
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, layout="constrained")
    for name, label in (
        ("rolling_low_amplitude_count_rmse", "Low: −50 ≤ V < 0 µV"),
        ("rolling_mid_amplitude_count_rmse", "Mid: −800 ≤ V < −50 µV"),
        ("rolling_high_amplitude_count_rmse", "High: V < −800 µV"),
        ("rolling_all_negative_amplitude_count_rmse", "All negative samples"),
    ):
        axes[0].plot(chunk_times_s, rolling_results[name], "o-", markersize=3, label=label)
    axes[0].set(title="Amplitude partition count stability", ylabel="RMSE (samples)")
    axes[1].plot(chunk_times_s, rolling_results["rolling_hist_change_rmse"], label="Unsmoothed")
    axes[1].plot(chunk_times_s, rolling_results["rolling_hist_change_rmse_smoothed"], label="Gaussian smoothed")
    axes[1].set(title="Full channel × amplitude-bin map", ylabel="RMSE (counts/bin)")
    axes[2].plot(chunk_times_s, rolling_results["rolling_hist_change_rmse_smoothed_log"], label="log10(1 + counts), then smoothed")
    axes[2].set(title="Full map after log transform", ylabel="RMSE (log units)", xlabel="Chunk midpoint (s, segment 0)")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Histogram stability — {baseline_label}")
    figures["04_histogram_stability"] = fig

    # Raw PSD curves: overlay every chunk for the same channel IDs, on log axes.
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.8 * rows), squeeze=False, layout="constrained")
    positive_freqs = freqs > 0
    psd_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(raw_chunk_times_s)))
    for ax, raw_index, channel_index in zip(axes.flat, raw_indices, selected_channels):
        for chunk_index, color in enumerate(psd_colors):
            values = psd[chunk_index, positive_freqs, raw_index]
            ax.loglog(freqs[positive_freqs], np.where(values > 0, values, np.nan), color=color, alpha=0.6, lw=0.8)
        ax.set_title(f"Channel {raw_recording.channel_ids[raw_index]} | depth {depths[channel_index]:g} µm")
        ax.set(xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[num_plot_channels:]:
        ax.set_visible(False)
    fig.suptitle("Raw AP power spectra — all chunks overlaid (DC omitted on log axis)")
    norm = plt.Normalize(float(raw_chunk_times_s.min()), float(raw_chunk_times_s.max()))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes.flat)[:num_plot_channels],
                 label="Chunk midpoint (s, segment 0)", shrink=0.7)
    figures["05_psd_channels"] = fig

    # PSD stability: complete power bands, with >10 Hz used only for 2D comparisons.
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, layout="constrained")
    for name, label in (
        ("rolling_low_frequency_power_rmse", "0–300 Hz"),
        ("rolling_mid_frequency_power_rmse", "300–6000 Hz"),
        ("rolling_high_frequency_power_rmse", "6000+ Hz"),
        ("rolling_all_frequency_power_rmse", "All frequencies"),
    ):
        axes[0].plot(raw_chunk_times_s, rolling_results[name], "o-", markersize=3, label=label)
    axes[0].set(title="Integrated band-power stability", ylabel="RMSE (µV²)")
    axes[1].plot(raw_chunk_times_s, rolling_results["rolling_psd_change_rmse"], label="Unsmoothed")
    axes[1].plot(raw_chunk_times_s, rolling_results["rolling_psd_change_rmse_smoothed"], label="Gaussian smoothed")
    axes[1].set(title="Full frequency × channel map (>10 Hz)", ylabel="RMSE (µV²/Hz)")
    axes[2].plot(raw_chunk_times_s, rolling_results["rolling_psd_change_rmse_smoothed_log"], label="log10(1 + PSD), then smoothed")
    axes[2].set(title="Full map after log transform (>10 Hz)", ylabel="RMSE (log units)", xlabel="Chunk midpoint (s, segment 0)")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle(f"PSD stability — {baseline_label}")
    figures["06_psd_stability"] = fig

    # Compare before/after CMR using identical chunk selections and color scales.
    for reference_index, (reference_key, reference_label) in enumerate((
        ("before_cmr", "Before CMR"), ("after_cmr", "After CMR"),
    )):
        xcorr_view = xcorr_results[reference_key]
        # Mean xcorr: use a fixed −1 to +1 color scale and channels ordered by depth.
        raw_depths = raw_recording.get_channel_locations()[:, 1]
        depth_order = np.argsort(raw_depths)
        correlation_cmap = plt.get_cmap("coolwarm").copy()
        correlation_cmap.set_bad("lightgray")
        fig = plt.figure(figsize=(11, 12), layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(grid[0])
        im = ax.imshow(xcorr_view["mean_matrix"][np.ix_(depth_order, depth_order)], origin="lower",
                       vmin=-1, vmax=1, cmap=correlation_cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(title=f"{reference_label}: mean zero-lag channel correlation | high-pass {xcorr_view['highpass_hz']:g} Hz\n"
                     f"{len(xcorr_view['matrices'])} random chunks, {xcorr_view['chunk_size_s']:g} s each")
        fig.colorbar(im, ax=ax, label="Pearson r")
        # Distance bars summarize all chunks, underneath the mean matrix.
        ax = fig.add_subplot(grid[1])
        distance_labels = ["third", "half", "two_thirds"]
        percentages = [xcorr_view["distance_results"][label]["high_percent"] for label in distance_labels]
        ax.bar(np.arange(3), percentages, color=["tab:blue", "tab:orange", "tab:green"])
        ax.set_xticks(np.arange(3), ["1/3", "1/2", "2/3"])
        ax.set(ylim=(0, 110), xlabel="Separation / recorded depth span (within channel group)",
               ylabel="Pair/chunk observations with high r (%)",
               title=f"All sampled chunks: r ≥ {xcorr_view['correlation_threshold']:g}; "
                     f"distance tolerance ±{xcorr_view['distance_tolerance_fraction']:g} of span")
        for position, (label, percentage) in enumerate(zip(distance_labels, percentages)):
            distance_summary = xcorr_view["distance_results"][label]
            num_pairs = distance_summary["num_pairs"]
            high_count = distance_summary["high_pair_chunk_count"]
            valid_count = distance_summary["valid_pair_chunk_count"]
            # Significant figures keep rare nonzero correlations visible.
            annotation = (
                f"{percentage:.3g}%\n{high_count:,} / {valid_count:,} observations\n{num_pairs:,} eligible pairs"
                if np.isfinite(percentage) else "No valid observations"
            )
            ax.text(position, percentage + 2 if np.isfinite(percentage) else 2, annotation, ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        figures[f"{7 + 2 * reference_index:02d}_xcorr_{reference_key}_mean_and_distances"] = fig

        # Random chunk examples have their own figure, without channel labels or ticks.
        selected_chunks = np.sort(np.random.default_rng(0).choice(
            len(xcorr_view["matrices"]), size=min(10, len(xcorr_view["matrices"])), replace=False
        ))
        matrix_rows = (len(selected_chunks) + 1) // 2
        fig = plt.figure(figsize=(12, 3.7 * matrix_rows), layout="constrained")
        grid = fig.add_gridspec(matrix_rows, 2)
        matrix_axes = []
        for plot_index, chunk_index in enumerate(selected_chunks):
            ax = fig.add_subplot(grid[plot_index // 2, plot_index % 2])
            matrix_axes.append(ax)
            im = ax.imshow(xcorr_view["matrices"][chunk_index][np.ix_(depth_order, depth_order)],
                           origin="lower", vmin=-1, vmax=1, cmap=correlation_cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Random sample #{chunk_index + 1}")
        fig.colorbar(im, ax=matrix_axes, label="Pearson r", shrink=0.8)
        fig.suptitle(f"{reference_label}: zero-lag correlations — random chunk examples")
        figures[f"{8 + 2 * reference_index:02d}_xcorr_{reference_key}_example_chunks"] = fig

    # Continuous AP RMS: all channels over consecutive windows, plus a channel summary.
    depth_order = np.argsort(depths)
    for segment_index, (values, times) in enumerate(zip(rolling_ap_rms["rms"], rolling_ap_rms["times"])):
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
        im = axes[0].pcolormesh(times, np.arange(len(depth_order)), values[:, depth_order].T,
                                shading="nearest", cmap="viridis", vmin=0)
        axes[0].set(ylabel="Channel rank by depth", title="All channels (increasing depth)")
        fig.colorbar(im, ax=axes[0], label="RMS (µV)")
        axes[1].plot(times, np.median(values, axis=1), label="Median across channels")
        axes[1].fill_between(times, np.percentile(values, 10, axis=1), np.percentile(values, 90, axis=1),
                             alpha=0.2, label="10th–90th percentile across channels")
        axes[1].set(xlabel=f"Chunk midpoint (recording time, s; segment {segment_index})", ylabel="RMS (µV)")
        axes[1].legend()
        axes[1].grid(alpha=0.2)
        fig.suptitle("Preprocessed AP rolling RMS — consecutive windows over the full recording")
        figures[f"11_ap_rolling_rms_segment_{segment_index}"] = fig

    # Population mean RMS trend: average channel RMS values within each sparse chunk.
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    slope = ap_mean_rms_linear_stability["slope_uV_per_s"]
    intercept = ap_mean_rms_linear_stability["intercept_uV"]
    r_squared = ap_mean_rms_linear_stability["r_squared"]
    ax.plot(chunk_times_s, ap_mean_rms_over_time, "o-", markersize=4, label="Mean RMS across channels")
    ax.plot(chunk_times_s, intercept + slope * chunk_times_s, "--", color="tab:red", label="Linear fit")
    ax.set(title=f"Preprocessed AP mean RMS trend (all channels)\nR² = {r_squared:.3f} | slope = {slope:.4g} µV/s",
           xlabel="Chunk midpoint (s, segment 0)", ylabel="Mean RMS (µV)")
    ax.legend()
    ax.grid(alpha=0.2)
    figures["12_ap_mean_rms_linear_fit"] = fig

    # One save/show switch controls every figure.
    if SAVE_PLOTS:
        PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = PLOT_FOLDER / f"{name}.png"
            fig.savefig(path, dpi=150)
            plot_files[name] = path
            plt.close(fig)
        print(f"Saved {len(plot_files)} plots to {PLOT_FOLDER}", flush=True)
    else:
        plt.show()
    return plot_files


def raw_data_quality_metrics(
    raw_ap_recording: BaseRecording,
    preprocessed_ap_recording: BaseRecording | None = None,
    first_chunk_time_s: float = 0.0,
    chunk_size_s: float = 1.0,
    num_chunks: int = 30,
    rolling_rms_window_size_s: float = 3.0,
    verbose: bool = False,
    **job_kwargs,
):
    """Compute sparse AP RMS, PSD, and histograms for debugger inspection."""
    # Explicit slices control sparse processing; configure a matching executor duration.
    sparse_job_kwargs, rolling_job_kwargs = _get_job_kwargs(job_kwargs, chunk_size_s, rolling_rms_window_size_s)

    if preprocessed_ap_recording is None:
        preprocessed_ap_recording = common_reference(highpass_filter(raw_ap_recording, freq_min=300), operator="median")

    # For each recording (and segment) generate the slices for the ChunkExecutor
    # to run over. These divide up each segment according to the chunk arguments.
    # Compute the RMS and PSDF over these sparsely samples chunks
    chunk_results = {}
    summaries = {}
    chunk_times = {}
    for recording_name, recording in zip(["raw", "prepro"], [raw_ap_recording, preprocessed_ap_recording]):
        if verbose:
            print(f"Running quality metrics for: {recording_name}", flush=True)

        sparse_slices_for_recording = _generate_skipped_chunk_slices(
            recording,
            first_chunk_time_s=first_chunk_time_s,
            chunk_size_s=chunk_size_s,
            num_chunks=num_chunks,
        )
        chunk_times[recording_name] = np.array([
            (start + end) / (2 * recording.get_sampling_frequency())
            for segment_index, start, end in sparse_slices_for_recording if segment_index == 0
        ])

        chunk_results[recording_name], summaries[recording_name] = _compute_rms_psd_histogram_over_sparse_chunks(
            recording, sparse_slices_for_recording, verbose, sparse_job_kwargs
        )

    # TODO: use MAD too

    # Inspect raw_ap_recording, preprocessed_ap_recording, chunk_results, and summaries here.
    ap_rms_array = np.stack(chunk_results["prepro"]["rms"][0])
    num_chunks = ap_rms_array.shape[0]
    num_chunks_for_baseline = 10

    # RMS
    # ------------------------------------------------------------------------------------------------------------------

    # RMS per channel
    ap_rms_per_channel = np.mean(ap_rms_array, axis=0)

    ap_median_rms = np.median(ap_rms_per_channel)

    # RMS CV per channel
    ap_rms_cv_per_channel = np.std(ap_rms_array, axis=0) / ap_rms_per_channel

    ap_median_rms_cv = np.median(ap_rms_cv_per_channel)

    # RMS linear stability
    from scipy.stats import linregress

    x = np.arange(num_chunks)

    result = linregress(x[:, None], ap_rms_array, axis=0)

    ap_rms_linear_stability = {
        "slopes": result.slope,
        "r_squared": result.rvalue ** 2,
        "p_values": result.pvalue
    }

    # Fit the mean of channel RMS values against actual sampled chunk times.
    ap_mean_rms_over_time = ap_rms_array.mean(axis=1)
    mean_rms_fit = linregress(chunk_times["prepro"], ap_mean_rms_over_time)
    ap_mean_rms_linear_stability = {
        "slope_uV_per_s": mean_rms_fit.slope,
        "intercept_uV": mean_rms_fit.intercept,
        "r_squared": mean_rms_fit.rvalue ** 2,
        "p_value": mean_rms_fit.pvalue,
    }

    # RMS drift across channels
    rolling_rms_change_rmse = _compute_rolling_rmse(ap_rms_array, num_chunks_for_baseline)
    # Actual RMS over consecutive windows (separate from sparse-chunk RMSE stability).
    rolling_ap_rms = _compute_rolling_rms(preprocessed_ap_recording, verbose, rolling_job_kwargs)

    # Histograms
    # ------------------------------------------------------------------------------------------------------------------

    ap_hist_array =  np.stack(chunk_results["prepro"]["hist"][0])
    ap_hist_bins = np.stack(chunk_results["prepro"]["shared_bins"])

    centers = (ap_hist_bins[:-1] + ap_hist_bins[1:]) / 2

    # Histogram partition
    low_bin_mask = (centers >= -50) & (centers < 0)
    mid_bin_mask = (centers >= -800) & (centers < -50)
    high_bin_mask = centers < -800

    # Preserve (chunks, channels) for sample-count stability over time.
    ap_low_amplitude_counts = ap_hist_array[:, :, low_bin_mask].sum(axis=-1)
    ap_mid_amplitude_counts = ap_hist_array[:, :, mid_bin_mask].sum(axis=-1)
    ap_high_amplitude_counts = ap_hist_array[:, :, high_bin_mask].sum(axis=-1)

    rolling_low_amplitude_count_rmse = _compute_rolling_rmse(ap_low_amplitude_counts, num_chunks_for_baseline)
    rolling_mid_amplitude_count_rmse = _compute_rolling_rmse(ap_mid_amplitude_counts, num_chunks_for_baseline)
    rolling_high_amplitude_count_rmse = _compute_rolling_rmse(ap_high_amplitude_counts, num_chunks_for_baseline)
    rolling_all_negative_amplitude_count_rmse = _compute_rolling_rmse(
        ap_low_amplitude_counts + ap_mid_amplitude_counts + ap_high_amplitude_counts, num_chunks_for_baseline
    )

    median_rolling_low_amplitude_count_rmse = np.median(rolling_low_amplitude_count_rmse)
    median_rolling_mid_amplitude_count_rmse = np.median(rolling_mid_amplitude_count_rmse)
    median_rolling_high_amplitude_count_rmse = np.median(rolling_high_amplitude_count_rmse)

    # compute rolling rmse over 2d num_chan x histogram, over chunks
    rolling_hist_change_rmse = _compute_rolling_rmse(ap_hist_array, num_chunks_for_baseline)

    # Sigma is in channel-index and amplitude-bin units, respectively.
    histogram_smoothing_sigma = (0.5, 0.5)
    rolling_hist_change_rmse_smoothed = _compute_rolling_rmse_2d(
        ap_hist_array, num_chunks_for_baseline, sigma=histogram_smoothing_sigma
    )
    rolling_hist_change_rmse_smoothed_log = _compute_rolling_rmse_2d(
        ap_hist_array, num_chunks_for_baseline, sigma=histogram_smoothing_sigma, log_base=10
    )

    # PSD
    # ------------------------------------------------------------------------------------------------------------------

    psd = np.stack(chunk_results["raw"]["psd"][0])
    freqs = np.asarray(chunk_results["raw"]["shared_freqs"])

    # PSD frequency partition
    low_freq_mask = (freqs >= 0) & (freqs < 300)
    mid_freq_mask = (freqs >= 300) & (freqs < 6000)
    high_freq_mask = freqs >= 6000
    frequency_bin_width = freqs[1] - freqs[0]

    # Integrate the uniformly spaced PSD density: (chunks, channels), in uV**2.
    ap_low_frequency_power = psd[:, low_freq_mask, :].sum(axis=1) * frequency_bin_width
    ap_mid_frequency_power = psd[:, mid_freq_mask, :].sum(axis=1) * frequency_bin_width
    ap_high_frequency_power = psd[:, high_freq_mask, :].sum(axis=1) * frequency_bin_width

    # Band-power stability uses each full frequency partition.
    rolling_low_frequency_power_rmse = _compute_rolling_rmse(
        ap_low_frequency_power, num_chunks_for_baseline
    )
    rolling_mid_frequency_power_rmse = _compute_rolling_rmse(
        ap_mid_frequency_power, num_chunks_for_baseline
    )
    rolling_high_frequency_power_rmse = _compute_rolling_rmse(
        ap_high_frequency_power, num_chunks_for_baseline
    )
    rolling_all_frequency_power_rmse = _compute_rolling_rmse(
        ap_low_frequency_power + ap_mid_frequency_power + ap_high_frequency_power, num_chunks_for_baseline
    )

    median_rolling_low_frequency_power_rmse = np.median(rolling_low_frequency_power_rmse)
    median_rolling_mid_frequency_power_rmse = np.median(rolling_mid_frequency_power_rmse)
    median_rolling_high_frequency_power_rmse = np.median(rolling_high_frequency_power_rmse)

    # Compare frequency x channel maps, discarding bins rather than zeroing them.
    stability_freq_mask = freqs > 10
    psd_for_stability = psd[:, stability_freq_mask, :]
    rolling_psd_change_rmse = _compute_rolling_rmse(psd_for_stability, num_chunks_for_baseline)

    # Sigma is in frequency-bin and channel-index units, respectively.
    psd_smoothing_sigma = (0.5, 0.5)
    rolling_psd_change_rmse_smoothed = _compute_rolling_rmse_2d(
        psd_for_stability, num_chunks_for_baseline, sigma=psd_smoothing_sigma
    )
    # Same log10(1 + data) convention as the histogram comparison.
    rolling_psd_change_rmse_smoothed_log = _compute_rolling_rmse_2d(
        psd_for_stability, num_chunks_for_baseline, sigma=psd_smoothing_sigma, log_base=10
    )

    # Channel cross-correlation (zero lag)
    # ------------------------------------------------------------------------------------------------------------------
    # TODO: Think more about how best to summarize shared high-frequency noise.
    xcorr_results_before_cmr = _compute_channel_xcorr(
        raw_ap_recording, num_chunks=100, chunk_size_s=1.0,
        highpass_hz=300, correlation_threshold=0.3,
        distance_tolerance_fraction=0.05, seed=0, reference=None,
    )
    xcorr_results_after_cmr = _compute_channel_xcorr(
        raw_ap_recording, num_chunks=100, chunk_size_s=1.0,
        highpass_hz=300, correlation_threshold=0.3,
        distance_tolerance_fraction=0.05, seed=0, reference="median",
    )
    xcorr_results = {
        "before_cmr": xcorr_results_before_cmr,
        "after_cmr": xcorr_results_after_cmr,
    }
    xcorr_mean_matrix_before_cmr = xcorr_results_before_cmr["mean_matrix"]
    xcorr_mean_matrix_after_cmr = xcorr_results_after_cmr["mean_matrix"]
    xcorr_std_matrix_before_cmr = xcorr_results_before_cmr["std_matrix"]
    xcorr_std_matrix_after_cmr = xcorr_results_after_cmr["std_matrix"]
    xcorr_distance_results_before_cmr = xcorr_results_before_cmr["distance_results"]
    xcorr_distance_results_after_cmr = xcorr_results_after_cmr["distance_results"]

    # Create Spreadsheet
    # ------------------------------------------------------------------------------------------------------------------

    # Keep the full time series as well as their scalar median summaries.
    rolling_results = {
        "rolling_rms_change_rmse": rolling_rms_change_rmse,
        "rolling_low_amplitude_count_rmse": rolling_low_amplitude_count_rmse,
        "rolling_mid_amplitude_count_rmse": rolling_mid_amplitude_count_rmse,
        "rolling_high_amplitude_count_rmse": rolling_high_amplitude_count_rmse,
        "rolling_all_negative_amplitude_count_rmse": rolling_all_negative_amplitude_count_rmse,
        "rolling_hist_change_rmse": rolling_hist_change_rmse,
        "rolling_hist_change_rmse_smoothed": rolling_hist_change_rmse_smoothed,
        "rolling_hist_change_rmse_smoothed_log": rolling_hist_change_rmse_smoothed_log,
        "rolling_low_frequency_power_rmse": rolling_low_frequency_power_rmse,
        "rolling_mid_frequency_power_rmse": rolling_mid_frequency_power_rmse,
        "rolling_high_frequency_power_rmse": rolling_high_frequency_power_rmse,
        "rolling_all_frequency_power_rmse": rolling_all_frequency_power_rmse,
        "rolling_psd_change_rmse": rolling_psd_change_rmse,
        "rolling_psd_change_rmse_smoothed": rolling_psd_change_rmse_smoothed,
        "rolling_psd_change_rmse_smoothed_log": rolling_psd_change_rmse_smoothed_log,
    }
    scalar_results = pd.DataFrame.from_dict(
        {
            "ap_median_rms": ap_median_rms,
            "ap_mean_rms_slope_uV_per_s": ap_mean_rms_linear_stability["slope_uV_per_s"],
            "ap_mean_rms_r_squared": ap_mean_rms_linear_stability["r_squared"],
            "ap_mean_rms_p_value": ap_mean_rms_linear_stability["p_value"],
            "ap_hist_underflow_percent": np.mean(chunk_results["prepro"]["hist_underflow_percent"][0]),
            "ap_hist_overflow_percent": np.mean(chunk_results["prepro"]["hist_overflow_percent"][0]),
            "raw_hist_underflow_percent": np.mean(chunk_results["raw"]["hist_underflow_percent"][0]),
            "raw_hist_overflow_percent": np.mean(chunk_results["raw"]["hist_overflow_percent"][0]),
            "xcorr_before_cmr_quarter_depth_high_percent": xcorr_distance_results_before_cmr["quarter"]["high_percent"],
            "xcorr_before_cmr_third_depth_high_percent": xcorr_distance_results_before_cmr["third"]["high_percent"],
            "xcorr_before_cmr_half_depth_high_percent": xcorr_distance_results_before_cmr["half"]["high_percent"],
            "xcorr_before_cmr_two_thirds_depth_high_percent": xcorr_distance_results_before_cmr["two_thirds"]["high_percent"],
            "xcorr_after_cmr_quarter_depth_high_percent": xcorr_distance_results_after_cmr["quarter"]["high_percent"],
            "xcorr_after_cmr_third_depth_high_percent": xcorr_distance_results_after_cmr["third"]["high_percent"],
            "xcorr_after_cmr_half_depth_high_percent": xcorr_distance_results_after_cmr["half"]["high_percent"],
            "xcorr_after_cmr_two_thirds_depth_high_percent": xcorr_distance_results_after_cmr["two_thirds"]["high_percent"],
            "ap_median_rms_cv": ap_median_rms_cv,
            "ap_median_rms_slope_uV_per_chunk": np.median(ap_rms_linear_stability["slopes"]),
            "ap_median_rms_r_squared": np.median(ap_rms_linear_stability["r_squared"]),
            "ap_median_rms_p_value": np.median(ap_rms_linear_stability["p_values"]),
            "median_rolling_rms_change_rmse": np.median(rolling_rms_change_rmse),
            "median_rolling_low_amplitude_count_rmse": median_rolling_low_amplitude_count_rmse,
            "median_rolling_mid_amplitude_count_rmse": median_rolling_mid_amplitude_count_rmse,
            "median_rolling_high_amplitude_count_rmse": median_rolling_high_amplitude_count_rmse,
            "median_rolling_all_negative_amplitude_count_rmse": np.median(rolling_all_negative_amplitude_count_rmse),
            "median_rolling_hist_change_rmse": np.median(rolling_hist_change_rmse),
            "median_rolling_hist_change_rmse_smoothed": np.median(rolling_hist_change_rmse_smoothed),
            "median_rolling_hist_change_rmse_smoothed_log": np.median(rolling_hist_change_rmse_smoothed_log),
            "median_rolling_low_frequency_power_rmse": median_rolling_low_frequency_power_rmse,
            "median_rolling_mid_frequency_power_rmse": median_rolling_mid_frequency_power_rmse,
            "median_rolling_high_frequency_power_rmse": median_rolling_high_frequency_power_rmse,
            "median_rolling_all_frequency_power_rmse": np.median(rolling_all_frequency_power_rmse),
            "median_rolling_psd_change_rmse": np.median(rolling_psd_change_rmse),
            "median_rolling_psd_change_rmse_smoothed": np.median(rolling_psd_change_rmse_smoothed),
            "median_rolling_psd_change_rmse_smoothed_log": np.median(rolling_psd_change_rmse_smoothed_log),
        },
        orient="index",
        columns=["value"],
    ).rename_axis("metric")

    # One row per channel. Counts and powers are means across sampled chunks.
    per_channel_results = pd.DataFrame(
        {
            "ap_rms_mean_uV": ap_rms_per_channel,
            "ap_rms_cv": ap_rms_cv_per_channel,
            "ap_rms_slope_uV_per_chunk": ap_rms_linear_stability["slopes"],
            "ap_rms_r_squared": ap_rms_linear_stability["r_squared"],
            "ap_rms_p_value": ap_rms_linear_stability["p_values"],
            "ap_low_amplitude_count_mean": ap_low_amplitude_counts.mean(axis=0),
            "ap_mid_amplitude_count_mean": ap_mid_amplitude_counts.mean(axis=0),
            "ap_high_amplitude_count_mean": ap_high_amplitude_counts.mean(axis=0),
            "ap_hist_underflow_percent": np.mean(chunk_results["prepro"]["hist_underflow_percent"][0], axis=0),
            "ap_hist_overflow_percent": np.mean(chunk_results["prepro"]["hist_overflow_percent"][0], axis=0),
        },
        index=pd.Index(preprocessed_ap_recording.channel_ids, name="channel_id"),
    )
    # Align raw PSD channels by ID in case preprocessing reordered the channels.
    per_channel_results = per_channel_results.join(
        pd.DataFrame(
            {
                "raw_power_0_300_Hz_mean_uV2": ap_low_frequency_power.mean(axis=0),
                "raw_power_300_6000_Hz_mean_uV2": ap_mid_frequency_power.mean(axis=0),
                "raw_power_6000_plus_Hz_mean_uV2": ap_high_frequency_power.mean(axis=0),
                "raw_hist_underflow_percent": np.mean(chunk_results["raw"]["hist_underflow_percent"][0], axis=0),
                "raw_hist_overflow_percent": np.mean(chunk_results["raw"]["hist_overflow_percent"][0], axis=0),
            },
            index=pd.Index(raw_ap_recording.channel_ids, name="channel_id"),
        ),
        how="outer",
        sort=False,
    )

    per_channel_results = per_channel_results.join(
        xcorr_results_before_cmr["per_channel_results"].add_prefix("before_cmr_"), how="outer", sort=False
    )
    per_channel_results = per_channel_results.join(
        xcorr_results_after_cmr["per_channel_results"].add_prefix("after_cmr_"), how="outer", sort=False
    )

    # Show / Save Plots
    # ------------------------------------------------------------------------------------------------------------------
    plot_files = _plot_quality_metrics(
        raw_ap_recording, preprocessed_ap_recording, chunk_times["prepro"], chunk_times["raw"],
        ap_rms_array, ap_rms_linear_stability, ap_hist_array, centers,
        psd, freqs, rolling_results, xcorr_results, num_chunks_for_baseline, rolling_ap_rms,
        ap_mean_rms_over_time, ap_mean_rms_linear_stability,
    )

    # Inspect scalar_results, per_channel_results, and rolling_results here.

    return {
        "chunk_results": chunk_results,
        "summaries": summaries,
        "scalar_results": scalar_results,
        "per_channel_results": per_channel_results,
        "rolling_results": rolling_results,
        "rolling_ap_rms": rolling_ap_rms,
        "ap_mean_rms_over_time": ap_mean_rms_over_time,
        "ap_mean_rms_linear_stability": ap_mean_rms_linear_stability,
        "xcorr_results": xcorr_results,
        "plot_files": plot_files,
    }







def _get_job_kwargs(job_kwargs, chunk_size_s, rolling_rms_window_size_s):
    """
    Get the job kwargs for the two chunk types"

    The passed `job_kwargs` should not contain options related to chunking, as
    we handle this outselves. Pass to `fix_job_kwargs`, for sparse_job_kwargs
    we generate ``slices`` outselves so this is not directly used, but passed
    to `fix_job_kwargs` to avoid it default inserting a chunk size.

    TODO: ensure this is properly tested, its a bit brittle.
    """
    chunk_keys = {"chunk_size", "chunk_duration", "chunk_memory", "total_memory"}
    if chunk_keys.intersection(job_kwargs):
        raise ValueError("Use the metric window parameters to control chunk sizes.")

    sparse_job_kwargs = fix_job_kwargs(
        {
            **job_kwargs,
            "chunk_duration": chunk_size_s,
        }
    )

    rolling_job_kwargs = fix_job_kwargs(
        {
            **job_kwargs,
            "chunk_duration": rolling_rms_window_size_s,
        }
    )

    return sparse_job_kwargs, rolling_job_kwargs


def _compute_rms_psd_histogram_over_sparse_chunks(recording, sparse_slices_for_recording, verbose, job_kwargs):
    """
    Compute the RMS metrics and PSD over uniformly sampled sparse chunks in parallel.

    The `TimeSeriesChunkExecutor` returns a list of tuple in the form (segment index, rms, ...)
    We first unnpack this to that all metrics are organised into separate arrays, by segment.
    The statistics (e.g. median, 10-90 quantiles) are computed over the chunks.
    """
    from spikeinterface.core import get_random_data_chunks

    chunks = get_random_data_chunks(
        recording,
        num_chunks_per_segment=2,
        chunk_size=max(1, int(recording.get_sampling_frequency() * 0.1)),
        seed=0,
        return_in_uV=True,
        concatenated=True,
    )

    # Across all sampled times and channels
    data_min = chunks.min()
    data_max = chunks.max()

    executor = TimeSeriesChunkExecutor(
        recording,
        _compute_rms_psd_histogram,
        _init_rms_worker,
        (recording, data_min, data_max),
        job_name="compute_rms",
        verbose=verbose,
        handle_returns=True,
        **job_kwargs,
    )
    results_per_chunk = executor.run(slices=sparse_slices_for_recording)

    # Unpack the (segment_index, rms, ...) per-chunk results into single arrays
    # organized by segment and result type
    num_segments = recording.get_num_segments()
    chunk_results = {
        "rms": [[] for _ in range(num_segments)],
        "psd": [[] for _ in range(num_segments)],
        "hist": [[] for _ in range(num_segments)],
        "hist_underflow_percent": [[] for _ in range(num_segments)],
        "hist_overflow_percent": [[] for _ in range(num_segments)],
    }

    shared_freqs = None
    shared_bins = None
    for segment_index, rms_by_channel, freqs, psd, hist, bins, underflow_percent, overflow_percent in results_per_chunk:
        chunk_results["rms"][segment_index].append(rms_by_channel)
        chunk_results["psd"][segment_index].append(psd)
        chunk_results["hist"][segment_index].append(hist)
        chunk_results["hist_underflow_percent"][segment_index].append(underflow_percent)
        chunk_results["hist_overflow_percent"][segment_index].append(overflow_percent)

        if shared_freqs is None:
            shared_freqs = freqs
        else:
            assert np.array_equal(shared_freqs, freqs), "All chunks should have the same frequency grid."

        if shared_bins is None:
            shared_bins = bins

    chunk_results["shared_freqs"] = shared_freqs
    chunk_results["shared_bins"] = shared_bins

    # Stack the per-chunk arrays and compute RMS summary statistics and
    # average the PSD over them
    summary_over_chunks = {
        "rms_median": [],
        "rms_quantile": [],
        "psd": [],
        "freqs": [],
        "hist": [],
    }
    for segment_index in range(num_segments):

        seg_rms_median = np.median(np.stack(chunk_results["rms"][segment_index]), axis=0)
        seg_rms_quantile = np.percentile(seg_rms_median, [10, 90])

        seg_welch_no_overlap = np.mean(
            np.stack(chunk_results["psd"][segment_index]),
            axis=0,
        )
        summary_over_chunks["rms_median"].append(seg_rms_median)
        summary_over_chunks["rms_quantile"].append(seg_rms_quantile)
        summary_over_chunks["psd"].append(seg_welch_no_overlap)
        summary_over_chunks["freqs"].append(shared_freqs)
        summary_over_chunks["hist"].append(np.sum(chunk_results["hist"][segment_index], axis=0))

    return chunk_results, summary_over_chunks


def _compute_rolling_rms(recording, verbose, job_kwargs):
    """
    Compute the 'rolling' RMS i.e. RMS of consecutive chunks of size ``rolling_rms_window_size_s``
    Note ``rolling_rms_window_size_s`` is set in on the job_kwargs in `get_job_kwargs`.

    Run in parallel and unpack the rseults into per-segment arrays for "rms" and "times",
    where "rms" is the rms computed over each chunk and "times" are the center time for each chunk.
    """
    executor = TimeSeriesChunkExecutor(
        recording,
        _compute_rms_with_times,
        _init_rms_worker,
        (recording, None, None),
        job_name="compute_rms_with_times",
        verbose=verbose,
        handle_returns=True,
        **job_kwargs,
    )
    results = executor.run()

    segment_results_chunked = {
        segment_index: {"rms": [], "times": []} for segment_index in range(recording.get_num_segments())
    }

    for segment_index, rms, time in results:
        segment_results_chunked[segment_index]["rms"].append(rms)
        segment_results_chunked[segment_index]["times"].append(time)

    return {
        "rms": [np.stack(data["rms"]) for data in segment_results_chunked.values()],
        "times": [np.asarray(data["times"]) for data in segment_results_chunked.values()],
    }


# Executor functions
# --------------------------------------------------------------------------------------

def _init_rms_worker(recording, data_min, data_max):
    return {
        "recording": recording,
        "data_min": data_min,
        "data_max": data_max
    }


def _compute_rms_psd_histogram(segment_index, start_frame, end_frame, worker_context):
    """
    Compute the RMS and PSD for an indivudal chunk.
    """
    recording = worker_context["recording"]
    data_min = worker_context["data_min"]
    data_max = worker_context["data_max"]

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_in_uV=True,
    ).astype(np.float64, copy=False)

    n_bins = 250
    # Cover each amplitude partition and put its thresholds on exact bin edges.
    # Extreme tails then stay in the correct partition when folded into edge bins.
    bins = np.unique(np.r_[
        np.linspace(min(data_min, -1000.0), max(data_max, 1.0), n_bins),
        -800.0, -50.0, 0.0,
    ])
    underflow_counts = np.sum(traces < bins[0], axis=0)
    overflow_counts = np.sum(traces > bins[-1], axis=0)
    underflow_percent = 100.0 * underflow_counts / traces.shape[0]
    overflow_percent = 100.0 * overflow_counts / traces.shape[0]
    hist = np.empty((traces.shape[1], bins.size - 1), dtype=np.int64)
    for chan_idx in range(traces.shape[1]):
        hist_counts, _ = np.histogram(traces[:, chan_idx], bins=bins)
        # Keep all finite samples: negative tail in first bin, positive tail in last.
        hist_counts[0] += underflow_counts[chan_idx]
        hist_counts[-1] += overflow_counts[chan_idx]
        hist[chan_idx, :] = hist_counts
    traces -= np.mean(traces, axis=0)
    rms = _compute_rms(traces)

    frequencies, psd = periodogram(
        traces,
        fs=recording.get_sampling_frequency(),
        window="hann",
        detrend="constant",
        scaling="density",
        axis=0,
    )

    return segment_index, rms, frequencies, psd, hist, bins, underflow_percent, overflow_percent


def _compute_rms_with_times(segment_index, start_frame, end_frame, worker_context):
    """
    Compute the RMS and time center for each chunk.
    """
    recording = worker_context["recording"]

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_in_uV=True,
    ).astype(np.float64, copy=False)

    traces -= np.mean(traces, axis=0)
    rms = _compute_rms(traces)

    rms_time_center_bin = (
        worker_context["recording"]
        .get_times(
            segment_index=segment_index,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        .mean()
    )

    return segment_index, rms, rms_time_center_bin


def _compute_rms(traces):
    return np.sqrt(np.mean(np.square(traces), axis=0))


# Slice generator
# --------------------------------------------------------------------------------------


def _generate_skipped_chunk_slices(
    recording,
    chunk_size_s,
    num_chunks,
    first_chunk_time_s=0.0,
):
    """Generate uniformly-spaced chunks times for each segment.

    The first chunk will start at ``first_chunk_time_s` and
    the last chunk will end on the last sample of the recording.
    The remaining chunks are then spread uniformly.
    """
    sampling_frequency = recording.get_sampling_frequency()
    chunk_size = round(chunk_size_s * sampling_frequency)
    first_frame = round(first_chunk_time_s * sampling_frequency)

    slices = []
    for segment_index in range(recording.get_num_segments()):

        num_samples = recording.get_num_samples(segment_index)
        max_start_frame = num_samples - chunk_size
        if max_start_frame < 0:
            raise ValueError(f"Segment {segment_index} is shorter than chunk_size_s={chunk_size_s}")

        # Divide the recording time into evenly spaced chunks with the first chunk starting
        # at `first_frame` and the last chunk finishing at example `num_samples`
        # Ensure that each segment has enough samples to fit all requested chunkds.
        available_samples = num_samples - first_frame
        if num_chunks * chunk_size > available_samples:
            raise ValueError(
                f"Segment {segment_index} is not long enough to accommodate the requested "
                "number of chunks at the requested chunk size."
            )

        start_frames = np.linspace(first_frame, max_start_frame, num_chunks)
        start_frames = np.rint(start_frames).astype(np.int64)

        slices.extend((segment_index, int(start_frame), int(start_frame + chunk_size)) for start_frame in start_frames)

    return slices


def main():
    # Edit these settings here; no command-line arguments are needed.
    recording_folder = Path(r"C:\Users\Jzimi\data\rdqm") / "recordings" / "ibl_np_5min"
    num_chunks = 40
    chunk_size_s = 1.0
    if not recording_folder.exists():
        raise FileNotFoundError(f"Recording folder does not exist: {recording_folder}")

    raw_ap_recording = load(recording_folder)
    print(
        f"Loaded {recording_folder}: {raw_ap_recording.get_num_channels()} channels, "
        f"{raw_ap_recording.get_num_samples()} samples",
        flush=True,
    )
    results = raw_data_quality_metrics(
        raw_ap_recording,
        num_chunks=num_chunks,
        chunk_size_s=chunk_size_s,
        verbose=True,
    )
    print("Raw RMS p10/p90 (uV):", results["summaries"]["raw"]["rms_quantile"][0], flush=True)
    print("Preprocessed RMS p10/p90 (uV):", results["summaries"]["prepro"]["rms_quantile"][0], flush=True)


if __name__ == "__main__":
    main()
