"""Sorting components: peak waveform features."""

from lib2to3.pgen2.token import AMPER
import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from ..toolkit import get_chunk_with_margin


def compute_waveform_features_peaks(
    recording,
    peaks,
    peak_sign="neg",
    time_range_list=[(0.2, 0.2), (1, 1.5)],
    feature_list=["amplitude", "ptps"],
    **job_kwargs,
):
    """Localize peak (spike) in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe
       * Y is axis 1 of the probe
       * Z is orthogonal to the plane of the probe

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    peaks: array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.
    peak_sign: str
        Sign of the template to compute best channels ('neg', 'pos', 'both')
    time_range_list: List
        List of time ranges around the peak to compute each feature.
    feature_list: List
        List of features to be computed
    {}

    Returns
    -------
    waveform_features: ndarray (NxCxF)
        Array with waveform features for each spike.
    """
    if not "amplitude" in feature_list and not "ptps" in feature_list:
        raise ValueError("features must be either amplitude or ptps")

    nbefore_max = 0
    nafter_max = 0
    if len(time_range_list) == 0 or len(feature_list) == 0:
        raise ValueError("You must pass in a non-empty feature_list or time_range_list")
    if len(time_range_list) != len(feature_list):
        raise ValueError("feature_list and time_range_list must be the same length")

    for i in range(len(time_range_list)):
        ms_before = time_range_list[i][0]
        ms_after = time_range_list[i][1]
        nbefore = int(ms_before * recording.get_sampling_frequency() / 1000.0)
        nafter = int(ms_after * recording.get_sampling_frequency() / 1000.0)
        if nbefore > nbefore_max:
            nbefore_max = nbefore
        if nafter > nafter_max:
            nafter_max = nafter
        time_range_list[i] = (nbefore, nafter)

    # margin at border for get_trace
    margin = max(nbefore_max, nafter_max)

    # TODO
    #  make a memmap for peaks to avoid serialization

    # and run
    func = _compute_waveform_features_chunk  # _localize_peaks_chunk
    init_func = _init_worker_compute_waveform_features_peaks
    init_args = (
        recording.to_dict(),
        peaks,
        margin,
        peak_sign,
        time_range_list,
        feature_list,
    )
    processor = ChunkRecordingExecutor(
        recording,
        func,
        init_func,
        init_args,
        handle_returns=True,
        job_name="compute waveform features",
        **job_kwargs,
    )
    peak_waveform_features = processor.run()
    peak_waveform_features = np.concatenate(peak_waveform_features)

    return peak_waveform_features


compute_waveform_features_peaks.__doc__ = (
    compute_waveform_features_peaks.__doc__.format(_shared_job_kwargs_doc)
)


def _init_worker_compute_waveform_features_peaks(
    recording,
    peaks,
    margin,
    peak_sign,
    time_range_list,
    feature_list,
):
    """Initialize worker for localizing peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor

        recording = load_extractor(recording)

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["peaks"] = peaks
    worker_ctx["margin"] = margin
    worker_ctx["feature_list"] = feature_list
    worker_ctx["time_range_list"] = time_range_list
    worker_ctx["peak_sign"] = peak_sign
    return worker_ctx


def _compute_waveform_features_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Localize peaks in a chunk of data."""

    # recover variables of the worker
    recording = worker_ctx["recording"]
    peaks = worker_ctx["peaks"]
    margin = worker_ctx["margin"]
    feature_list = worker_ctx["feature_list"]
    peak_sign = worker_ctx["peak_sign"]
    time_range_list = worker_ctx["time_range_list"]

    # load trace in memory
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(
        recording_segment, start_frame, end_frame, None, margin, add_zeros=True
    )

    # get local peaks (sgment + start_frame/end_frame)
    i0 = np.searchsorted(peaks["segment_ind"], segment_index)
    i1 = np.searchsorted(peaks["segment_ind"], segment_index + 1)
    peak_in_segment = peaks[i0:i1]
    i0 = np.searchsorted(peak_in_segment["sample_ind"], start_frame)
    i1 = np.searchsorted(peak_in_segment["sample_ind"], end_frame)
    local_peaks = peak_in_segment[i0:i1]

    # make sample index local to traces
    local_peaks = local_peaks.copy()
    local_peaks["sample_ind"] -= start_frame - left_margin

    peak_waveform_features = compute_waveform_features(
        traces,
        local_peaks,
        peak_sign,
        time_range_list,
        feature_list,
    )

    return peak_waveform_features


def compute_waveform_features(
    traces, local_peak, peak_sign, time_range_list, feature_list
):
    """Localize peaks using the center of mass method."""
    peak_waveform_features = np.empty(
        (local_peak.size, traces.shape[1], len(feature_list))
    )
    do_amplitudes = False
    do_ptps = False
    idx = 0
    for feature in feature_list:
        if feature == "amplitude":
            do_amplitudes = True
            amplitude_idx = idx
            idx += 1
        if feature == "ptps":
            do_ptps = True
            ptp_idx = idx
            idx += 1

    for i, peak in enumerate(local_peak):
        if do_amplitudes:
            nbefore = time_range_list[amplitude_idx][0]
            nafter = time_range_list[amplitude_idx][1]
            wf = traces[peak["sample_ind"] - nbefore : peak["sample_ind"] + nafter, :]
            if peak_sign == "both":
                wf_amplitude = np.max(np.abs(wf, axis=0))
            elif peak_sign == "neg":
                wf_amplitude = np.min(wf, axis=0)
            elif peak_sign == "pos":
                wf_amplitude = np.max(wf, axis=0)
            peak_waveform_features[i, :, amplitude_idx] = wf_amplitude
        if do_ptps:
            nbefore = time_range_list[ptp_idx][0]
            nafter = time_range_list[ptp_idx][1]
            wf = traces[peak["sample_ind"] - nbefore : peak["sample_ind"] + nafter, :]
            wf_ptp = wf.ptp(axis=0)
            peak_waveform_features[i, :, ptp_idx] = wf_ptp
    return peak_waveform_features
