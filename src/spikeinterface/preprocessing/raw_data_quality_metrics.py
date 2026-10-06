import csv
from pathlib import Path

import numpy as np
from scipy.signal import periodogram

from spikeinterface.core import BaseRecording
from spikeinterface.core.job_tools import fix_job_kwargs, TimeSeriesChunkExecutor

from .common_reference import common_reference
from .filter import highpass_filter


def raw_data_quality_metrics(
    raw_ap_recording: BaseRecording,
    raw_lfp_recording: BaseRecording | None = None,
    preprocessed_ap_recording: BaseRecording | None = None,
    first_chunk_time_s: float = 0.0,
    chunk_size_s: float = 1.0,
    num_chunks: int = 30,
    rolling_rms_window_size_s: float = 3.0,
    folder: Path | str | None = None,
    verbose: bool = False,
    **job_kwargs,
):
    """Compute raw data quality metrics for the provided recordings.

    Based on the IBL white paper (reference below), this function computes a set of
    metrics used to assess raw data quality metrics. First, it takes uniformly sampled
    chunks of data across the recording and computes the RMS and hann-windowed periodogram
    for each. Then it computes the median RMS across time (per channel), the 10-90
    quantiles across channels and averages across periodograms, yielding a non-overlapping
    Welch PSD per channel. The IBL use 1s every 300s, here we use 1s for 30 chunks over the recording.
    See 'Returns' section for details.

    Next, a 'rolling' RMS is computed using consecutive chunks across the entire
    recording. This is used to generate plots rather than compute metrics.

    All quality metrics are computed on the raw recording, the preprocessed recording and
    LFP recording (if provided). If no AP-preprocessed recording is used, the data is filtered
    and common median referenced.

    Parameters
    ---------

    raw_ap_recording :
        The raw AP-band recording (i.e. has not been preprocessed).
    raw_lfp_recording :
        The unprocessed LFP-band recording.
    preprocessed_ap_recording :
        The preprocessed AP-band recording. This can be included to use a recording
        preprocessed to your steps, otherwise if ``None`` a standard 300 Hz highpass
        filter and common median referencing is used.
    first_chunk_time_s :
        The time of the first chunk for non-consecutive uniform sampling used
        for RMS median, quantiles and PSD computation.
    chunk_size_s :
        The size of the chunks for non-consecutive uniform sampling used
        for RMS median, quantiles and PSD computation. Uses the IBLs 1s default.
    num_chunks :
        The number of chunks for non-consecutive uniform sampling used
        for RMS median, quantiles and PSD computation.
    rolling_rms_window_size_s : float, default: 3.0
        The window size used for the consecutive windows across which
        the 'rolling' RMS is computed.
    folder :
        If provided, write one CSV per recording type in this directory. Each CSV
        has one row per channel for median RMS and a separate ``all_channels`` row
        for the 10th and 90th percentiles. Multi-segment columns have a segment suffix.
        Array-valued results remain in the return value.
    verbose :
        Whether to print the progress of the quality metric computation.
    **job_kwargs : dict
        Keyword arguments forwarded to :class:`TimeSeriesChunkExecutor` (e.g. ``n_jobs``).
        This must not contain arguments related to chunk-size, as these are set by this module.

    Returns
    ------

    A dictionary with keys ``"chunked_results"`` and ``"rolling_rms"``.

    ``"chunked_results"" :
        A dictionary with keys [recording_name][metric][segment_index]. The
        recording names are "raw_ap_recording", "preprocessed_ap_recording" and (if LFP
        provided) "raw_lfp_recording". The metric names are:

        "rms_median" :
            The median of the sampled chunk RMS values, per channel (num_channels,).
            This gives a typical RMS value for each channel. Higher is worse.
        "rms_quantile" :
            The 10th and 90th percentiles of `rms_median` across channels, shape (2,).
            This gives a measure of variance of the RMS across channels (it is assumed
            that channel RMS should all be similar). Larger spread is worse.
        "psd" :
            The average PSD computed across the chunks, shape (num_freqs, num_channels). Each chunk is Hann-windowed
            and the periodogram computed. These periodograms are averaged into the "psd".
            This is like a Welch PSD computed over uniformly sampled, non-overlapping chunks.
        "freqs" :
            The frequency bins associated with "psd", size (num_freqs,).

    ``"rolling_rms"``
        A dictionary with keys [recording_name]["rms" or "times"][segment_index]
        where "rms" gives the rolling RMS across the recording and "times" gives the
        center time for each corresponding RMS measurement.

    References
    ----------
    International Brain Laboratory. "Spike sorting pipeline for the International
    Brain Laboratory." https://doi.org/10.6084/m9.figshare.19705522

    """
    # Explicit slices control sparse processing; configure a matching executor duration.
    sparse_job_kwargs, rolling_job_kwargs = _get_job_kwargs(job_kwargs, chunk_size_s, rolling_rms_window_size_s)

    if not np.isfinite(first_chunk_time_s) or first_chunk_time_s < 0:
        raise ValueError("first_chunk_time_s must be finite and non-negative")

    if preprocessed_ap_recording is None:
        preprocessed_ap_recording = highpass_filter(raw_ap_recording, freq_min=300)
        preprocessed_ap_recording = common_reference(preprocessed_ap_recording, operator="median")

    recordings = {
        "raw_ap_recording": raw_ap_recording,
        "preprocessed_ap_recording": preprocessed_ap_recording,
    }

    if raw_lfp_recording is not None:
        recordings.update({"raw_lfp_recording": raw_lfp_recording})

    all_chunk_results = {}
    all_rolling_rms = {}

    for recording_name, recording in recordings.items():

        if verbose:
            print(f"Running quality metrics for: {recording_name}")

        # For each recording (and segment) generate the slices for the ChunkExecutor
        # to run over. These divide up each segment according to the chunk arguments.
        sparse_slices_for_recording = _generate_skipped_chunk_slices(
            recording,
            first_chunk_time_s=first_chunk_time_s,
            chunk_size_s=chunk_size_s,
            num_chunks=num_chunks,
        )

        # Compute the RMS and PSDF over these sparsely samples chunks
        chunk_results = _compute_rms_psd_over_sparse_chunks(
            recording, sparse_slices_for_recording, verbose, sparse_job_kwargs
        )

        # Compute the RMS over the entire recording in blocks
        # of size `rolling_rms_window_size_s`.
        rolling_rms = _compute_rolling_rms(recording, verbose, rolling_job_kwargs)

        all_chunk_results[recording_name] = chunk_results
        all_rolling_rms[recording_name] = rolling_rms

    if folder is not None:
        _write_rms_summary_csvs(folder, recordings, all_chunk_results)
        raise NotImplementedError("Plots saving is not yet implemented.")

    return {
        "chunked_results": all_chunk_results,
        "rolling_rms": all_rolling_rms,
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


def _compute_rms_psd_over_sparse_chunks(recording, sparse_slices_for_recording, verbose, job_kwargs):
    """
    Compute the RMS metrics and PSD over uniformly sampled sparse chunks in parallel.

    The `TimeSeriesChunkExecutor` returns a list of tuple in the form (segment index, rms, ...)
    We first unnpack this to that all metrics are organised into separate arrays, by segment.
    The statistics (e.g. median, 10-90 quantiles) are computed over the chunks.
    """
    executor = TimeSeriesChunkExecutor(
        recording,
        _compute_rms_psd,
        _init_rms_worker,
        (recording,),
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
    }

    shared_freqs = None
    for segment_index, rms_by_channel, freqs, psd in results_per_chunk:
        chunk_results["rms"][segment_index].append(rms_by_channel)
        chunk_results["psd"][segment_index].append(psd)

        if shared_freqs is None:
            shared_freqs = freqs
        else:
            assert np.array_equal(shared_freqs, freqs), "All chunks should have the same frequency grid."

    # Stack the per-chunk arrays and compute RMS summary statistics and
    # average the PSD over them
    summary_over_chunks = {
        "rms_median": [],
        "rms_quantile": [],
        "psd": [],
        "freqs": [],
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

    return summary_over_chunks


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
        (recording,),
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


def _init_rms_worker(recording):
    return {
        "recording": recording,
    }


def _compute_rms_psd(segment_index, start_frame, end_frame, worker_context):
    """
    Compute the RMS and PSD for an indivudal chunk.
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

    frequencies, psd = periodogram(
        traces,
        fs=recording.get_sampling_frequency(),
        window="hann",
        detrend="constant",
        scaling="density",
        axis=0,
    )

    return segment_index, rms, frequencies, psd


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


def _write_rms_summary_csvs(folder, recordings, chunked_results):
    """Write the metrics to a CSV which has rms_median per channel (row),
    as well as the 10-90 quantiles in the first row of separate columns"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for recording_name, recording in recordings.items():
        metrics = chunked_results[recording_name]
        num_segments = len(metrics["rms_median"])
        columns = ["channel_id"]
        channel_rows = [{"channel_id": channel_id} for channel_id in recording.get_channel_ids()]
        summary_row = {"channel_id": "all_channels"}

        for segment_index in range(num_segments):
            suffix = f"_seg_{segment_index}" if num_segments > 1 else ""
            median_column = f"rms_median{suffix}"
            p10_column = f"rms_quantile_10{suffix}"
            p90_column = f"rms_quantile_90{suffix}"
            columns.extend((median_column, p10_column, p90_column))

            for channel_index, row in enumerate(channel_rows):
                row[median_column] = metrics["rms_median"][segment_index][channel_index]

            summary_row[p10_column], summary_row[p90_column] = metrics["rms_quantile"][segment_index]

        with (folder / f"{recording_name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerow(summary_row)
            writer.writerows(channel_rows)


# ----------------------------------------------------------------------------------------------------------------------
# Plot Raw
# ----------------------------------------------------------------------------------------------------------------------


# TODO: LLM WROTE THIS FOR PROTOTYPING. NEEDS REWRITING.
def plot_raw_data_quality_metrics(chunk_results, rolling_rms, segment_index=0, *, recordings=None):
    """Plot one segment, optionally ordering channels by probe depth.

    ``recordings`` maps result names to the recordings used to compute them.
    When supplied, channels with locations are ordered using
    ``order_channels_by_depth`` (tip first, bottom of the plot), and ticks show
    channel IDs. Otherwise, rows retain their original recording order.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from spikeinterface.core import order_channels_by_depth

    recording_names = ["raw_ap_recording", "preprocessed_ap_recording"]
    if "raw_lfp_recording" in chunk_results or "raw_lfp_recording" in rolling_rms:
        recording_names.append("raw_lfp_recording")
    missing_names = [name for name in recording_names if name not in chunk_results or name not in rolling_rms]
    if missing_names:
        raise ValueError(f"Missing results for recordings: {', '.join(missing_names)}")
    if recordings is not None and any(name not in recordings for name in recording_names):
        raise ValueError("recordings must include each recording represented in the results")

    num_segments = len(rolling_rms["raw_ap_recording"]["rms"])
    if num_segments == 0:
        raise ValueError("At least one recording segment is required")
    for name in recording_names:
        for results, metrics in (
            (rolling_rms[name], ("rms", "times")),
            (chunk_results[name], ("rms_median", "rms_quantile", "psd", "freqs")),
        ):
            if any(metric not in results or len(results[metric]) != num_segments for metric in metrics):
                raise ValueError(f"All metrics for {name} must contain {num_segments} segments")
    if not isinstance(segment_index, (int, np.integer)) or not 0 <= segment_index < num_segments:
        raise IndexError(f"segment_index must be an integer between 0 and {num_segments - 1}, got {segment_index}")

    # Validate all selected data before creating figures.
    selected = {}
    for name in recording_names:
        data = {
            metric: np.asarray(chunk_results[name][metric][segment_index])
            for metric in ("rms_median", "rms_quantile", "psd", "freqs")
        }
        data.update({metric: np.asarray(rolling_rms[name][metric][segment_index]) for metric in ("rms", "times")})
        for value_key, coordinate_key in (("rms", "times"), ("psd", "freqs")):
            values, coordinates = data[value_key], data[coordinate_key]
            if values.ndim != 2 or coordinates.ndim != 1 or values.size == 0 or values.shape[0] != coordinates.size:
                raise ValueError(f"Invalid {value_key}/{coordinate_key} shapes for {name} segment {segment_index}")
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{value_key} for {name} must contain finite, non-negative values")
            if not np.all(np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0):
                raise ValueError(f"{coordinate_key} for {name} must be finite and strictly increasing")
        num_channels = data["rms"].shape[1]
        if (
            data["psd"].shape[1] != num_channels
            or data["rms_median"].shape != (num_channels,)
            or data["rms_quantile"].shape != (2,)
        ):
            raise ValueError(f"Inconsistent channel or RMS summary shapes for {name} segment {segment_index}")
        if np.any(data["freqs"] < 0):
            raise ValueError(f"Frequencies for {name} must be non-negative")
        for metric in ("rms_median", "rms_quantile"):
            if not np.all(np.isfinite(data[metric])) or np.any(data[metric] < 0):
                raise ValueError(f"{metric} for {name} must contain finite, non-negative values")
        data["channel_ids"] = np.arange(num_channels)
        data["channel_label"] = "Channel index (recording order)"
        if recordings is not None:
            recording = recordings[name]
            if recording.get_num_channels() != num_channels:
                raise ValueError(f"Recording channel count does not match results for {name}")
            order = np.arange(num_channels)
            data["channel_label"] = "Channel ID (recording order)"
            if recording.has_channel_location():
                if not np.all(np.isfinite(recording.get_channel_locations())):
                    raise ValueError(f"Channel locations for {name} must be finite to order by depth")
                order, _ = order_channels_by_depth(recording)
                data["channel_label"] = "Channel ID (depth ordered)"
            data["channel_ids"] = recording.get_channel_ids()[order]
            data["rms"] = data["rms"][:, order]
            data["psd"] = data["psd"][:, order]
            data["rms_median"] = data["rms_median"][order]
        selected[name] = data

    def label_channels(axis, data):
        channel_ids = data["channel_ids"]
        ticks = np.linspace(0, channel_ids.size - 1, min(8, channel_ids.size), dtype=int)
        axis.set_yticks(ticks, labels=[str(channel_ids[index]) for index in ticks])
        axis.set_ylabel(data["channel_label"])

    def plot_values(axis, coordinates, values, norm=None):
        num_channels = values.shape[1]
        if coordinates.size == 1:
            # A single center gives no window width; show values at that coordinate.
            image = axis.scatter(
                np.full(num_channels, coordinates[0]),
                np.arange(num_channels),
                c=values[0],
                marker="s",
                norm=norm,
            )
        else:
            # Infer display edges from centers, retaining irregular spacing at the final window.
            edges = np.r_[
                coordinates[0] - (coordinates[1] - coordinates[0]) / 2,
                coordinates[:-1] + np.diff(coordinates) / 2,
                coordinates[-1] + (coordinates[-1] - coordinates[-2]) / 2,
            ]
            image = axis.pcolormesh(
                edges,
                np.arange(num_channels + 1) - 0.5,
                values.T,
                shading="flat",
                norm=norm,
            )
        axis.set_ylim(-0.5, num_channels - 0.5)
        return image

    display_names = {
        "raw_ap_recording": "Raw AP",
        "preprocessed_ap_recording": "Preprocessed AP",
        "raw_lfp_recording": "LFP",
    }
    num_recordings = len(recording_names)
    rms_figure, rms_axes = plt.subplots(1, num_recordings, figsize=(5 * num_recordings, 3.5), constrained_layout=True)
    for axis, name in zip(rms_axes, recording_names):
        data = selected[name]
        image = plot_values(axis, data["times"], data["rms"])
        rms_quantiles = data["rms_quantile"]
        axis.set_title(
            f"{display_names[name]} | segment {segment_index}\n"
            f"median RMS {np.median(data['rms_median']):.2f} \u00b5V; "
            f"channel P10\u2013P90 {rms_quantiles[0]:.2f}\u2013{rms_quantiles[1]:.2f} \u00b5V",
            fontsize=10,
        )
        axis.set_xlabel("Time (s)")
        label_channels(axis, data)
        rms_figure.colorbar(image, ax=axis, label="RMS (\u00b5V)")

    spectrum_figure, spectrum_axes = plt.subplots(
        1, num_recordings, figsize=(5 * num_recordings, 3.5), constrained_layout=True
    )
    for axis, name in zip(spectrum_axes, recording_names):
        data = selected[name]
        psd = data["psd"]
        positive_psd = psd[psd > 0]
        if positive_psd.size:
            psd_floor = positive_psd.min()
            psd_ceiling = positive_psd.max()
            if psd_ceiling == psd_floor:
                psd_floor /= 10
            psd_for_plot = np.maximum(psd, psd_floor)
            norm = LogNorm(vmin=psd_floor, vmax=psd_ceiling)
        else:
            # Zero power has no logarithm; keep zeros visible on a linear color scale.
            psd_for_plot = psd
            norm = Normalize(vmin=0, vmax=1)
        image = plot_values(axis, data["freqs"], psd_for_plot, norm=norm)
        if data["freqs"].size > 1:
            axis.set_xlim(left=0)
        axis.set_title(f"{display_names[name]} | segment {segment_index}")
        axis.set_xlabel("Frequency (Hz)")
        label_channels(axis, data)
        spectrum_figure.colorbar(image, ax=axis, label="PSD (\u00b5V\u00b2/Hz)")

    return rms_figure, spectrum_figure
