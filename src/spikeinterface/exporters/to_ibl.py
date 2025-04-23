from __future__ import annotations

from importlib.util import find_spec
import re
import shutil
import warnings
from pathlib import Path

import numpy as np

from spikeinterface.core import SortingAnalyzer, BaseRecording, get_random_data_chunks
from spikeinterface.core.job_tools import fix_job_kwargs, ChunkRecordingExecutor, _shared_job_kwargs_doc
from spikeinterface.core.template_tools import get_template_extremum_channel
from spikeinterface.exporters import export_to_phy


def export_to_ibl_gui(
    sorting_analyzer: SortingAnalyzer,
    output_folder: str | Path,
    lfp_recording: BaseRecording | None = None,
    rms_win_length_s=3,
    welch_win_length_samples=2**14,
    psd_chunk_duration_s=1,
    psd_num_chunks=100,
    good_units_query: str | None = "amplitude_median < -40 and isi_violations_ratio < 0.5 and amplitude_cutoff < 0.2",
    remove_if_exists: bool = False,
    verbose: bool = True,
    **job_kwargs,
):
    """
    Exports a sorting analyzer to the format required by the `IBL alignment GUI <https://github.com/int-brain-lab/iblapps/wiki>`_.

    Parameters
    ----------
    analyzer: SortingAnalyzer
        The sorting analyzer object to use for spike information.
        Should also contain the pre-processed recording to use for AP-band data.
    output_folder: str | Path
        The output folder for the exports.
    lfp_recording: BaseRecording | None, default: None
        The pre-processed recording to use for LFP data. If None, the LFP data is not exported.
    rms_win_length_s: float, default: 3
        The window length in seconds for the RMS calculation (on the LFP data).
    welch_win_length_samples: int, default: 2^14
        The window length in samples for the Welch spectral density computation (on the LFP data).
    psd_chunk_duration_s: float, default: 1
        The chunk duration in seconds for the spectral density calculation (on the LFP data).
    psd_num_chunks: int, default: 100
        The number of chunks to use for the spectral density calculation (on the LFP data).
    remove_if_exists: bool, default: False
        If True and "output_folder" exists, it is removed and overwritten
    verbose: bool, default: True
        If True, output is verbose

    """

    if find_spec("scipy") is None:
        raise ImportError("Please install scipy to use the `export_to_ibl` function.")
    else:
        from scipy.signal import welch

    if sorting_analyzer.get_num_segments() != 1:
        raise ValueError("The export to IBL format only supports a single segment.")

    # Output folder checks
    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f"{output_folder} already exists")

    if verbose:
        print("Exporting recording to IBL format...")

    # Compute any missing extensions
    required_extensions = [
        "templates",
        "spike_amplitudes",
        "quality_metrics",
    ]
    for ext in required_extensions:
        if not sorting_analyzer.has_extension(ext):
            raise ValueError(f"Missing required extension: {ext}. Please compute it before exporting to IBL format.")

    # Check in case user pre-calculated a small set of qm's that aren't enough for IBL
    if good_units_query is not None:
        quality_metrics_in_query = re.split(">|<|>=|<=|==|and", good_units_query)[::2]
        required_qms = [qm_name.strip() for qm_name in quality_metrics_in_query]
        qm = sorting_analyzer.get_extension("quality_metrics").get_data()
        missing_metrics = []
        for qm_name in required_qms:
            if qm_name not in qm.columns:
                missing_metrics.append(qm_name)
        if len(missing_metrics) > 0:
            raise ValueError(
                f"Missing required quality metrics: {missing_metrics}. Please compute it before exporting to IBL format."
            )

    # Make sure output dir exists, in case user skips export_to_phy
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True, exist_ok=True)

    ### Save spikes info ###
    extremum_channel_indices = get_template_extremum_channel(sorting_analyzer, outputs="index")
    spikes = sorting_analyzer.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_indices)

    # spikes.clusters
    np.save(output_folder / "spikes.clusters.npy", spikes["unit_index"].astype("int32"))

    # spike depths
    if sorting_analyzer.has_extension("spike_locations"):
        spike_locations = sorting_analyzer.get_extension("spike_locations").get_data()
        spike_depths = spike_locations["y"]
    else:
        # we use the extremum channel depth for each spike
        spike_depths = sorting_analyzer.get_channel_locations()[:, 1][spikes["channel_index"]]
    np.save(output_folder / "spikes.depths.npy", spike_depths.astype("float32"))

    # spike times
    spike_sample_indices = spikes["sample_index"]
    if sorting_analyzer.has_recording() and sorting_analyzer.recording.has_time_vector():
        spike_times = sorting_analyzer.recording.get_times()[spike_sample_indices]
    else:
        spike_times = spike_sample_indices / sorting_analyzer.sampling_frequency
    np.save(output_folder / "spikes.times.npy", spike_times.astype("float64"))

    # spike amps
    amps = sorting_analyzer.get_extension("spike_amplitudes").get_data()
    amps_positive_in_V = -amps * 1e-6
    np.save(output_folder / "spikes.amps.npy", amps_positive_in_V.astype("float32"))

    ### Save clusters info ###

    # templates
    templates = sorting_analyzer.get_extension("templates").get_data()
    np.save(output_folder / "clusters.waveforms.npy", templates)

    # cluster channels
    extremum_channel_indices = get_template_extremum_channel(sorting_analyzer, outputs="index")
    cluster_channels = np.array(list(extremum_channel_indices.values()), dtype="int32")
    np.save(output_folder / "clusters.channels.npy", cluster_channels)

    # peak-to-trough durations

    # if template_metrics are already computed, use them to get the peak-to-trough durations
    peak_to_trough_durations = None
    if sorting_analyzer.has_extension("template_metrics"):
        template_metrics = sorting_analyzer.get_extension("template_metrics").get_data()
        if "peak_to_valley" in template_metrics.columns:
            peak_to_trough_durations = template_metrics["peak_to_valley"].values

    # if not, we will compute them ourselves
    if peak_to_trough_durations is None:
        peak_to_trough_durations = []
        # get the channel index of the max amplitude for each cluster
        for unit_index, unit_id in enumerate(sorting_analyzer.unit_ids):
            template = templates[unit_index, :, :]
            extremum_channel_index = extremum_channel_indices[unit_id]
            peak_waveform = template[:, extremum_channel_index]
            peak_to_trough = (np.argmax(peak_waveform) - np.argmin(peak_waveform)) / sorting_analyzer.sampling_frequency
            peak_to_trough_durations.append(peak_to_trough)
        peak_to_trough_durations = np.array(peak_to_trough_durations)
    np.save(output_folder / "clusters.peakToTrough.npy", peak_to_trough_durations)

    # quality metrics
    qm = sorting_analyzer.get_extension("quality_metrics")
    qm_data = qm.get_data()
    qm_data.index.name = "cluster_id"
    qm_data["cluster_id.1"] = qm_data.index.values

    if good_units_query is None:
        qm_data["label"] = 1
    else:
        good_units = qm_data.query(good_units_query)
        good_units_indices = good_units.index.values
        labels = np.zeros(len(qm_data), dtype="int32")
        qm_data["label"] = labels
        qm_data.loc[good_units_indices, "label"] = 1
    qm_data.to_csv(output_folder / "clusters.metrics.csv")

    ### Save channels info ###

    # channel positions
    channel_positions = sorting_analyzer.get_channel_locations()
    np.save(output_folder / "channels.localCoordinates.npy", channel_positions)

    # channel indices
    np.save(output_folder / "channels.rawInd.npy", np.arange(sorting_analyzer.get_num_channels(), dtype="int32"))

    # Now we need to add the extra IBL specific files
    # See here for docs on the format: https://github.com/int-brain-lab/iblapps/wiki/3.-Overview-of-datasets#input-histology-data
    if sorting_analyzer.has_recording():
        # Get RMS for the preprocessed (AP) data. We will use a window of length rms_win_length_s seconds slid over the entire recording.
        if verbose:
            print("Computing AP RMS")
        recording_ap = sorting_analyzer.recording
        job_kwargs_ = job_kwargs.copy()
        job_kwargs_["chunk_duration"] = f"{rms_win_length_s}s"
        rms_preprocessed, rms_times = compute_rms(
            recording_ap,
            verbose=verbose,
            **job_kwargs_,
        )
        np.save(output_folder / "_iblqc_ephysTimeRmsAP.rms.npy", rms_preprocessed)
        np.save(output_folder / "_iblqc_ephysTimeRmsAP.timestamps.npy", rms_times)
    elif verbose:
        print("No recording data found in the SortingAnalyzer, skipping AP RMS calculation.")

    if lfp_recording is not None:
        # Get RMS for the LFP data
        if verbose:
            print("Computing LFP RMS")
        job_kwargs_ = job_kwargs.copy()
        job_kwargs_["chunk_duration"] = f"{rms_win_length_s}s"
        rms_lfp, rms_times = compute_rms(lfp_recording, verbose=verbose, **job_kwargs_)
        np.save(output_folder / "_iblqc_ephysTimeRmsLF.rms.npy", rms_lfp)
        np.save(output_folder / "_iblqc_ephysTimeRmsLF.timestamps.npy", rms_times)

        # Get spectral density on a snippet of LFP data
        if verbose:
            print("Computing LFP PSD")
        lfp_sample_data = get_random_data_chunks(
            lfp_recording,
            num_chunks_per_segment=psd_num_chunks,
            chunk_duration=f"{psd_chunk_duration_s}s",
            return_scaled=True,
            concatenated=True,
        )
        psd = np.zeros((welch_win_length_samples // 2 + 1, lfp_sample_data.shape[1]), dtype=np.float32)
        for i_channel in range(lfp_sample_data.shape[1]):
            freqs, Pxx = welch(
                lfp_sample_data[:, i_channel],
                fs=lfp_recording.sampling_frequency,
                nperseg=welch_win_length_samples,
            )
            psd[:, i_channel] = Pxx
        freqs = freqs.astype(np.float32)
        np.save(output_folder / "_iblqc_ephysSpectralDensityLF.power.npy", psd)
        np.save(output_folder / "_iblqc_ephysSpectralDensityLF.freqs.npy", freqs)


def compute_rms(
    recording: BaseRecording,
    verbose: bool = False,
    **job_kwargs,
):
    """
    Compute the RMS of a recording in chunks.

    Parameters
    ----------
    recording: BaseRecording
        The recording object to compute the RMS for.
    {}
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    # use executor (loop or workers)
    func = _compute_rms_chunk
    init_func = _init_rms_worker
    init_args = (recording,)
    executor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        job_name="compute_rms",
        verbose=verbose,
        handle_returns=True,
        **job_kwargs,
    )
    results = executor.run()

    rms_values = np.zeros((len(results), recording.get_num_channels()))
    rms_times = np.zeros((len(results)))

    for i, result in enumerate(results):
        rms_values[i, :], rms_times[i] = result

    return rms_values, rms_times


def _init_rms_worker(recording):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["times"] = recording.get_times()
    return worker_ctx


def _compute_rms_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx["recording"]
    times = worker_ctx["times"]

    traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    rms = np.sqrt(np.mean(traces**2, axis=0))
    # get the middle time of the chunk
    if end_frame < recording.get_num_samples() - 1:
        middle_frame = (start_frame + end_frame) // 2
    else:
        # if we are at the end of the recording, use the middle point of the last chunk
        middle_frame = (start_frame + recording.get_num_samples() - 1) // 2
    # get the time of the middle frame
    rms_time = times[middle_frame]

    return rms, rms_time


compute_rms.__doc__ = compute_rms.__doc__.format(_shared_job_kwargs_doc)
