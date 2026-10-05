import numpy as np
from scipy.signal import periodogram

from spikeinterface.core import BaseRecording
from spikeinterface.core.job_tools import fix_job_kwargs, TimeSeriesChunkExecutor

from .common_reference import common_reference
from .filter import highpass_filter, lowpass_filter

# but whether RMS should exclude DC must be an explicit metric decision.
# do we want to add some saturation detection?
# TODO: rename these vars
# factor out the chunk running stuff
# think about the format ouf outputs (last)
# Reduction matches IBL. Differences remain: sampling starts at 0 vs 40 s, chunks are demeaned,
# processed AP uses high-pass/common-reference instead of destriping, and values are µV rather than IBL’s volts.
# how to handle last chunk when num_samples // num_samples has remainder
# log axis for rms / psd
#      if num_chunks < 1:
#         raise ValueError("num_chunks_per_segment must be at least 1")
#    if num_chunks > max_start_frame + 1:
#       raise ValueError(f"Segment {segment_index} is too short for {num_chunks} distinct chunk positions")
# TODO: review spike rate, this isn't so useful in current form may have missunderstood, recheck IBL!
# TODO: num_chunks estimate option?


def raw_data_quality_metrics(
    raw_ap_recording: BaseRecording,
    raw_lfp_recording: BaseRecording | None = None,
    preprocessed_ap_recording: BaseRecording | None = None,
    chunk_time_s=1.0,
    num_chunks=24,
    verbose: bool = False,
    **job_kwargs,
):
    """ """
    job_kwargs = fix_job_kwargs(job_kwargs)

    if preprocessed_ap_recording is None:
        preprocessed_ap_recording = highpass_filter(raw_ap_recording, freq_min=300)
        preprocessed_ap_recording = common_reference(preprocessed_ap_recording, operator="median")

    recordings = {
        "raw": raw_ap_recording,
        "prepro_ap": preprocessed_ap_recording,
    }

    if raw_lfp_recording is not None:
        recordings.update({"lfp": raw_lfp_recording})

    all_chunk_results = {}
    all_rolling_rms = {}

    for recording_name, recording in recordings.items():

        sparse_slices_for_recording = _generate_skipped_chunk_slices(
            recording,
            chunk_time_s=chunk_time_s,
            num_chunks=num_chunks,
        )

        chunk_results = _compute_rms_psd_over_sparse_chunks(recording, sparse_slices_for_recording, verbose, job_kwargs)

        rolling_rms = _compute_rolling_rms(recording, verbose, job_kwargs)

        all_chunk_results[recording_name] = chunk_results
        all_rolling_rms[recording_name] = rolling_rms

    return {
        "quick": all_chunk_results,
        "rolling_rms": all_rolling_rms,
    }


def _compute_rms_psd_over_sparse_chunks(recording, sparse_slices_for_recording, verbose, job_kwargs):
    """"""
    resuts_per_chunk = _run_executor_over_chunks(
        recording,
        _compute_rms_psd,
        job_name="compute_rms",
        verbose=verbose,
        job_kwargs=job_kwargs,
        slices=sparse_slices_for_recording,
    )

    # Unpack the results into per-segment arrays
    segment_results_chunked = {
        segment_index: {"rms": [], "psd": [], "freqs": None} for segment_index in range(recording.get_num_segments())
    }

    for segment_index, rms_by_channel, freqs, psd in resuts_per_chunk:
        segment_results_chunked[segment_index]["rms"].append(rms_by_channel)
        segment_results_chunked[segment_index]["psd"].append(psd)

        if segment_results_chunked[segment_index]["freqs"] is None:
            segment_results_chunked[segment_index]["freqs"] = freqs

    # Stack the per-chunk arrays and compute summary statistics over them
    summary_over_chunks = {
        "rms_median_over_chunk": [],
        "rms_quantile_over_channel": [],
        "welch_no_overlap": [],
        "freqs": [],
    }

    for segment_index, data in segment_results_chunked.items():

        rms_median_over_chunk = np.median(np.stack(data["rms"]), axis=0)
        rms_quantile_over_channel = np.percentile(rms_median_over_chunk, [10, 90])

        welch_no_overlap = np.mean(
            np.stack(data["psd"]),
            axis=0,
        )
        summary_over_chunks["rms_median_over_chunk"].append(rms_median_over_chunk)
        summary_over_chunks["rms_quantile_over_channel"].append(rms_quantile_over_channel)
        summary_over_chunks["welch_no_overlap"].append(welch_no_overlap)
        summary_over_chunks["freqs"].append(data["freqs"])

    return summary_over_chunks


def _compute_rolling_rms(recording, verbose, job_kwargs):
    """"""
    results = _run_executor_over_chunks(
        recording,
        _compute_rms_with_times,
        job_name="compute_rms_with_times",
        verbose=verbose,
        job_kwargs=job_kwargs,
        slices=None,
    )

    segment_results_chunked = {
        segment_index: {"rms": [], "times": []} for segment_index in range(recording.get_num_segments())
    }

    for segment_index, rms, time in results:
        segment_results_chunked[segment_index]["rms"].append(rms)
        segment_results_chunked[segment_index]["times"].append(time)

    final_results = [
        {
            "rms": np.stack(data["rms"]),
            "times": np.asarray(data["times"]),
        }
        for data in segment_results_chunked.values()
    ]

    return final_results


def _compute_rms_psd(segment_index, start_frame, end_frame, worker_context):
    """"""
    recording = worker_context["recording"]

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_in_uV=True,
    ).astype(np.float64, copy=False)

    rms = _compute_rms(traces)

    frequencies, psd = periodogram(
        traces,
        fs=recording.get_sampling_frequency(),
        window="hann",  # TODO: expose this
        detrend="constant",
        scaling="density",
        axis=0,
    )

    return segment_index, rms, frequencies, psd


def _compute_rms_with_times(segment_index, start_frame, end_frame, worker_context):
    """"""
    recording = worker_context["recording"]

    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_in_uV=True,
    ).astype(np.float64, copy=False)

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


def _init_rms_worker(recording):
    return {
        "recording": recording,
    }


def _run_executor_over_chunks(
    recording,
    function_to_run,
    job_name,
    verbose,
    job_kwargs,
    slices,
):
    executor = TimeSeriesChunkExecutor(
        recording,
        function_to_run,
        _init_rms_worker,
        (recording,),
        job_name=job_name,
        verbose=verbose,
        handle_returns=True,
        **job_kwargs,
    )

    return executor.run(slices=slices)


def _generate_skipped_chunk_slices(
    recording,
    chunk_time_s,
    num_chunks,
):
    """"""
    sampling_frequency = recording.get_sampling_frequency()
    chunk_size = round(chunk_time_s * sampling_frequency)

    slices = []
    for segment_index in range(recording.get_num_segments()):

        num_samples = recording.get_num_samples(segment_index)
        max_start_frame = num_samples - chunk_size
        if max_start_frame < 0:
            # Reject short segments explicitly: negative starts can produce unequal chunk lengths
            # and incompatible PSD frequency grids when the per-chunk spectra are stacked.
            raise ValueError(f"Segment {segment_index} is shorter than chunk_time_s={chunk_time_s}")

        # Uniformly spaced starts still overlap (or repeat) unless all requested chunks fit.
        if num_chunks * chunk_size > num_samples:
            raise ValueError(
                f"Segment {segment_index} has {num_samples} samples, but {num_chunks} non-overlapping "
                f"chunks of {chunk_size} samples require {num_chunks * chunk_size} samples. "
                "Reduce num_chunks or chunk_time_s."
            )

        num_chunks_for_segment = num_chunks

        if num_chunks_for_segment == 1:
            start_frames = np.array([0], dtype=np.int64)
        else:
            start_to_start_interval_size = max_start_frame / (num_chunks_for_segment - 1)
            start_frames = np.rint(np.arange(num_chunks_for_segment) * start_to_start_interval_size)

        slices.extend((segment_index, int(start_frame), int(start_frame + chunk_size)) for start_frame in start_frames)

    return slices


# ----------------------------------------------------------------------------------------------------------------------
# Plot Raw
# ----------------------------------------------------------------------------------------------------------------------


def plot_raw_data_quality_metrics(chunk_results, rolling_rms, segment_index=0):
    """Plot rolling RMS and power spectra for one segment.
    TODO: rewrite
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    recording_names = ("raw", "prepro_ap", "lfp")
    missing_names = [name for name in recording_names if name not in chunk_results or name not in rolling_rms]
    if missing_names:
        raise ValueError(f"Missing results for recordings: {', '.join(missing_names)}")

    num_segments = len(rolling_rms["raw"])
    if num_segments == 0:
        raise ValueError("At least one recording segment is required")
    if any(len(rolling_rms[name]) != num_segments for name in recording_names):
        raise ValueError("All rolling RMS results must have the same number of segments")
    if not 0 <= segment_index < num_segments:
        raise IndexError(f"segment_index must be between 0 and {num_segments - 1}, got {segment_index}")

    display_names = {"raw": "Raw AP", "prepro_ap": "Preprocessed AP", "lfp": "LFP"}
    rms_figure, rms_axes = plt.subplots(1, 3, figsize=(15, 3.5), constrained_layout=True)
    for axis, recording_name in zip(rms_axes, recording_names):
        segment_data = rolling_rms[recording_name][segment_index]
        rms = np.asarray(segment_data["rms"])
        times = np.asarray(segment_data["times"])
        if rms.ndim != 2 or times.ndim != 1 or rms.shape[0] != times.size:
            raise ValueError(
                f"Invalid rolling RMS shapes for {recording_name} segment {segment_index}: "
                f"rms={rms.shape}, times={times.shape}"
            )

        image = axis.imshow(
            rms.T,
            origin="lower",
            aspect="auto",
            extent=(times[0], times[-1], 0, rms.shape[1]),
            interpolation="nearest",
        )
        rms_by_channel = np.asarray(chunk_results[recording_name]["rms_median_over_chunk"][segment_index])
        rms_quantiles = np.asarray(chunk_results[recording_name]["rms_quantile_over_channel"][segment_index])
        axis.set_title(
            f"{display_names[recording_name]} | segment {segment_index}\n"
            f"median RMS {np.median(rms_by_channel):.2f} µV; "
            f"channel P10–P90 {rms_quantiles[0]:.2f}–{rms_quantiles[1]:.2f} µV",
            fontsize=10,
        )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Channel index")
        rms_figure.colorbar(image, ax=axis, label="RMS (µV)")

    spectrum_figure, spectrum_axes = plt.subplots(1, 3, figsize=(15, 3.5), constrained_layout=True)
    for axis, recording_name in zip(spectrum_axes, recording_names):
        frequencies = np.asarray(chunk_results[recording_name]["freqs"][segment_index])
        psd = np.asarray(chunk_results[recording_name]["welch_no_overlap"][segment_index])
        if psd.ndim != 2 or frequencies.ndim != 1 or psd.shape[0] != frequencies.size:
            raise ValueError(
                f"Invalid PSD shapes for {recording_name} segment {segment_index}: "
                f"psd={psd.shape}, frequencies={frequencies.shape}"
            )
        positive_psd = psd[psd > 0]
        psd_floor = positive_psd.min() if positive_psd.size else np.finfo(float).tiny
        psd_for_plot = np.maximum(psd, psd_floor)
        psd_ceiling = psd_for_plot.max()
        if psd_ceiling <= psd_floor:
            psd_ceiling = psd_floor * 10
        image = axis.imshow(
            psd_for_plot.T,
            origin="lower",
            aspect="auto",
            extent=(frequencies[0], frequencies[-1], 0, psd.shape[1]),
            interpolation="nearest",
            norm=LogNorm(vmin=psd_floor, vmax=psd_ceiling),
        )
        axis.set_title(f"{display_names[recording_name]} | segment {segment_index}")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Channel index")
        spectrum_figure.colorbar(image, ax=axis, label="PSD (µV²/Hz)")

    return rms_figure, spectrum_figure
