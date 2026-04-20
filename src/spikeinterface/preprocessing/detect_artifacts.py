from typing import Literal

import numpy as np

from spikeinterface.core import BaseRecording
from spikeinterface.core.base import base_period_dtype
from spikeinterface.preprocessing.rectify import RectifyRecording
from spikeinterface.preprocessing.common_reference import CommonReferenceRecording
from spikeinterface.preprocessing.filter_gaussian import GaussianFilterRecording
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.recording_tools import get_noise_levels, get_random_data_chunks
from spikeinterface.core.node_pipeline import PeakDetector, base_peak_dtype, run_node_pipeline, PipelineNode

artifact_dtype = base_period_dtype

# this will be extend with channel boundaries if needed
# extended_artifact_dtype = artifact_dtype + [
#     # TODO
# ]


def _collapse_events(events: np.ndarray) -> np.ndarray:
    """
    Collapse artifact events that were split across chunk boundaries.

    When a chunk boundary falls within an artifact period the period is emitted
    as two adjacent events whose ``end_sample_index`` / ``start_sample_index``
    values are equal.  This function merges such pairs into a single record.

    Parameters
    ----------
    events : np.ndarray
        Array of artifact events with dtype ``artifact_dtype``, containing
        ``"start_sample_index"``, ``"end_sample_index"``, and
        ``"segment_index"`` fields.

    Returns
    -------
    np.ndarray
        Array of collapsed artifact events with the same dtype as ``events``.
    """
    order = np.lexsort((events["start_sample_index"], events["segment_index"]))
    events = events[order]
    to_drop = np.zeros(events.size, dtype=bool)

    # compute if duplicate
    for i in np.arange(events.size - 1):
        # We use the + 1 because the end sample index is inclusive
        # so if the next start sample index is exactly 1 more than the end sample index,
        # then they are part of the same artifact period
        overlapping = events["end_sample_index"][i] + 1 >= events["start_sample_index"][i + 1]
        same_segment = events["segment_index"][i] == events["segment_index"][i + 1]
        if overlapping and same_segment:
            to_drop[i] = True
            events["start_sample_index"][i + 1] = events["start_sample_index"][i]
    collapsed_events = events[~to_drop]
    return collapsed_events


## detect_period_artifacts_saturation zone
class _DetectSaturation(PipelineNode):
    """
    A pipeline node for parallelised amplifier-saturation detection.

    When run with :func:`run_node_pipeline`, this node computes saturation
    events for a given data chunk.  See :func:`detect_saturation_periods` for
    the full algorithm description and parameter semantics.
    """

    name = "detect_saturation"
    preferred_mp_context = None
    _compute_has_extended_signature = True

    def __init__(
        self,
        recording: BaseRecording,
        saturation_threshold_uV: float,
        diff_threshold_uV: float | None,
        proportion: float,
    ) -> None:
        """
        Parameters
        ----------
        recording : BaseRecording
            The recording to process.
        saturation_threshold_uV : float
            Voltage saturation threshold in μV.
        diff_threshold_uV : float | None
            First-derivative threshold in μV/sample, or ``None`` to disable
            derivative-based detection.
        proportion : float
            Fraction of channels that must exceed the threshold for a sample to
            be labelled as saturated (0 < proportion < 1).
        """
        PipelineNode.__init__(self, recording, return_output=True)

        num_chans = recording.get_num_channels()

        self.diff_threshold_uV = diff_threshold_uV
        thresh = np.full((num_chans,), saturation_threshold_uV)
        # 0.98 is empirically determined as the true saturating point is
        # slightly lower than the documented saturation point of the probe
        self.sampling_frequency = recording.get_sampling_frequency()
        self.proportion = proportion
        self._dtype = np.dtype(artifact_dtype)
        self.gain = recording.get_channel_gains()
        self.offset = recording.get_channel_offsets()

        self.saturation_threshold_unscaled = (thresh - self.offset) / self.gain

        # do not apply offset when dealing with the derivative
        if self.diff_threshold_uV is not None:
            self.diff_threshold_unscaled = diff_threshold_uV / self.gain
        else:
            self.diff_threshold_unscaled = None

    def get_margin(self) -> int:
        """Return the number of margin samples required on each side of a chunk."""
        return 0

    def get_dtype(self) -> np.dtype:
        """Return the NumPy dtype of the output array produced by :meth:`compute`."""
        return self._dtype

    def compute(
        self,
        traces: np.ndarray,
        start_frame: int,
        end_frame: int,
        segment_index: int,
        max_margin: int,
    ) -> tuple[np.ndarray]:
        """
        Detect saturation events within a single chunk of raw traces.

        A sample is labelled as *saturated by value* when the fraction of
        channels whose absolute amplitude exceeds
        ``saturation_threshold_unscaled`` is greater than ``proportion``.

        Optionally, a sample is also labelled as *saturated by derivative* when
        the fraction of channels whose forward-difference amplitude exceeds
        ``diff_threshold_unscaled`` is greater than ``proportion``.

        Consecutive saturated samples are grouped into contiguous period events.

        Parameters
        ----------
        traces : np.ndarray
            Raw trace data for the current chunk, shape ``(n_samples, n_channels)``.
        start_frame : int
            Index of the first sample of this chunk within its segment.
        end_frame : int
            Index one past the last sample of this chunk within its segment.
        segment_index : int
            Index of the segment to which this chunk belongs.
        max_margin : int
            Maximum trace margin (unused; kept for API compatibility).

        Returns
        -------
        tuple[np.ndarray]
            A one-element tuple containing an array of saturation events with
            dtype ``artifact_dtype``.
        """
        # cast to float32 to prevent overflow when applying thresholds in unscaled ADC units
        traces = traces.astype("float32")

        saturation = np.mean(np.abs(traces) > self.saturation_threshold_unscaled, axis=1)
        detected_by_value = saturation > self.proportion

        if self.diff_threshold_unscaled is not None:
            # then compute the derivative of the voltage saturation
            n_diff_saturated = np.mean(np.abs(np.diff(traces, axis=0)) >= self.diff_threshold_unscaled, axis=1)

            # Note this means the velocity is not checked for the last sample in the
            # check because we are taking the forward derivative
            n_diff_saturated = np.r_[n_diff_saturated, 0]

            # if either of those reaches more than the proportion of channels labels the sample as saturated
            detected_by_diff = n_diff_saturated > self.proportion
            saturation = np.logical_or(detected_by_value, detected_by_diff)
        else:
            saturation = detected_by_value

        intervals = np.flatnonzero(np.diff(saturation, prepend=False, append=False))
        n_events = len(intervals) // 2  # Number of saturation periods
        events = np.zeros(n_events, dtype=artifact_dtype)

        for i, (start, stop) in enumerate(zip(intervals[::2], intervals[1::2])):
            events[i]["start_sample_index"] = start + start_frame
            events[i]["end_sample_index"] = stop + start_frame
            events[i]["segment_index"] = segment_index

        return (events,)


def detect_saturation_periods(
    recording: BaseRecording,
    saturation_threshold_uV: float | None = None,
    diff_threshold_uV: float | None = None,
    proportion: float = 0.2,
    job_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Detect amplifier saturation events (single- or multi-sample periods) in raw data.

    Saturation detection should be applied to the **raw** recording, before any
    preprocessing.  The returned periods can then be used to zero out (silence)
    the corresponding samples **after** preprocessing has been performed.

    Saturation is identified in two complementary ways:

    1. **By value**: a sample is saturated when the fraction of channels whose
       absolute amplitude exceeds ``saturation_threshold_uV`` is greater than
       ``proportion``.
    2. **By derivative**: a sample is saturated when the fraction of channels
       whose forward-difference amplitude exceeds ``diff_threshold_uV`` is
       greater than ``proportion``.

    If ``diff_threshold_uV`` is not ``None``, a sample is marked as saturated
    if *either* criterion is met.

    Parameters
    ----------
    recording : BaseRecording
        The recording on which to detect saturation events.
    saturation_threshold_uV : float | None, default: None
        Voltage saturation threshold in μV.  The appropriate value depends on
        the probe and amplifier gain settings; for Neuropixels 1.0 probes IBL
        recommend **1200 μV**.  NP2 probes are harder to saturate than NP1.
        If ``None``, the value is read from the ``"saturation_threshold_uV"``
        annotation of ``recording``.
    diff_threshold_uV : float | None, default: None
        First-derivative threshold in μV/sample.  Periods where the
        sample-to-sample voltage change exceeds this value in the required
        fraction of channels are flagged as saturation.  Pass ``None`` to
        disable derivative-based detection and rely solely on
        ``saturation_threshold_uV``.  IBL use **300 μV/sample** for NP1 probes.
    proportion : float, default: 0.2
        Fraction of channels (0 < proportion < 1) that must exceed the
        threshold for a sample to be considered saturated.
    job_kwargs : dict | None, default: None
        Keyword arguments forwarded to :func:`run_node_pipeline` (e.g.
        ``n_jobs``, ``chunk_duration``).

    Returns
    -------
    np.ndarray
        Array with dtype ``artifact_dtype`` describing each saturation period.
        Fields: ``"start_sample_index"``, ``"end_sample_index"``,
        ``"segment_index"``.
    """
    if job_kwargs is None:
        job_kwargs = {}

    job_kwargs = fix_job_kwargs(job_kwargs)

    # The saturation threshold can be specified in the recording annotations and loaded automatically
    # for some acquisition systems (e.g., Neuropixels)
    if "saturation_threshold_uV" in recording.get_annotation_keys() and saturation_threshold_uV is None:
        saturation_threshold_uV = recording.get_annotation("saturation_threshold_uV")

    if saturation_threshold_uV is None:
        raise ValueError(
            "Cannot read `saturation_threshold_uV` from recording. Please pass `saturation_threshold_uV` manually."
        )

    node0 = _DetectSaturation(
        recording,
        saturation_threshold_uV=saturation_threshold_uV,
        diff_threshold_uV=diff_threshold_uV,
        proportion=proportion,
    )

    saturation_periods = run_node_pipeline(
        recording, [node0], job_kwargs=job_kwargs, job_name="detect saturation artifacts", check_for_peak_source=False
    )
    return _collapse_events(saturation_periods)


## detect_artifact_periods_by_envelope zone
class _DetectThresholdCrossing(PeakDetector):
    """
    A pipeline node that detects threshold crossings of a channel-aggregated envelope.

    Each crossing of the global median z-score above 1 is returned as an event
    with a ``"front"`` flag indicating whether the crossing is a rising edge
    (``True``) or a falling edge (``False``).  Used internally by
    :func:`detect_artifact_periods_by_envelope`.

    Attributes
    ----------
    abs_thresholds : np.ndarray
        Per-channel absolute amplitude thresholds in raw ADC units.
    """

    name = "threshold_crossings"
    preferred_mp_context = None

    def __init__(
        self,
        recording: BaseRecording,
        mads: np.ndarray,
        medians: np.ndarray,
        detect_threshold: float = 5,
    ) -> None:
        """
        Parameters
        ----------
        recording : BaseRecording
            The (pre-processed envelope) recording to process.
        detect_threshold : float, default: 5
            Detection threshold expressed as a multiple of the estimated noise
            level per channel.
        mads : np.ndarray
            Pre-computed per-channel median absolute deviations in raw ADC units.
        medians : np.ndarray
            Pre-computed per-channel medians in raw ADC units.
        noise_levels_kwargs : dict, default: {}
            Additional keyword arguments forwarded to
            :func:`~spikeinterface.core.get_noise_levels`.
        """
        PeakDetector.__init__(self, recording, return_output=True)
        self.abs_thresholds = (mads * detect_threshold)[np.newaxis, :]
        self.medians = medians[np.newaxis, :]
        # internal dtype
        self._dtype = np.dtype([("sample_index", "int64"), ("segment_index", "int64"), ("front", "bool")])

    def get_margin(self) -> int:
        """Return the number of margin samples required on each side of a chunk."""
        return 0

    def get_dtype(self) -> np.dtype:
        """Return the NumPy dtype of the output array produced by :meth:`compute`."""
        return self._dtype

    def compute(
        self,
        traces: np.ndarray,
        start_frame: int,
        end_frame: int,
        segment_index: int,
        max_margin: int,
    ) -> tuple[np.ndarray]:
        """
        Detect threshold crossings in a single chunk of envelope traces.

        The per-sample signal is the median z-score across channels:
        ``z = median(traces / abs_thresholds, axis=1)``.  Transitions of
        ``z > 1`` are located and returned as crossing events.

        Parameters
        ----------
        traces : np.ndarray
            Envelope trace data for the current chunk,
            shape ``(n_samples, n_channels)``.
        start_frame : int
            Index of the first sample of this chunk within its segment.
        end_frame : int
            Index one past the last sample of this chunk within its segment.
        segment_index : int
            Index of the segment to which this chunk belongs.
        max_margin : int
            Maximum trace margin (unused; kept for API compatibility).

        Returns
        -------
        tuple[np.ndarray]
            A one-element tuple containing an array of threshold-crossing
            events with fields ``"sample_index"``, ``"segment_index"``, and
            ``"front"`` (``True`` for rising edge, ``False`` for falling edge).
        """
        z = np.median((traces - self.medians) / self.abs_thresholds, axis=1)
        threshold_mask = np.diff((z > 1) != 0, axis=0)

        indices = np.flatnonzero(threshold_mask)
        threshold_crossings = np.zeros(indices.size, dtype=self._dtype)
        threshold_crossings["sample_index"] = indices
        threshold_crossings["segment_index"] = segment_index
        threshold_crossings["front"][::2] = True
        threshold_crossings["front"][1::2] = False
        return (threshold_crossings,)


def detect_artifact_periods_by_envelope(
    recording: BaseRecording,
    detect_threshold: float = 5,
    apply_envelope_common_reference: bool = False,
    freq_max: float = 20.0,
    seed: int | None = None,
    job_kwargs: dict | None = None,
    random_slices_kwargs: dict | None = None,
    return_envelope: bool = False,
) -> np.ndarray | tuple[np.ndarray, BaseRecording]:
    """
    Detect putative artifact periods as threshold crossings of a global channel envelope.

    The pipeline is:

    1. Rectify the raw recording.
    2. Low-pass filter with a Gaussian filter up to ``freq_max`` Hz to produce
       a smooth per-channel amplitude envelope.
    3. Apply a common-average reference so that only signals correlated across
       channels (i.e. artefacts) survive.
    4. Estimate per-channel noise levels on the envelope.
    5. Detect samples where the median channel z-score exceeds
       ``detect_threshold``, and convert contiguous runs into period records.

    Parameters
    ----------
    recording : BaseRecording
        The recording extractor from which to detect artefact periods.
    detect_threshold : float, default: 5
        Detection threshold as a multiple of the estimated per-channel noise
        level of the envelope.
    freq_max : float, default: 20.0
        Cut-off frequency (Hz) for the Gaussian low-pass filter applied to the
        rectified signal when building the envelope.
    seed : int | None, default: None
        Random seed forwarded to :func:`~spikeinterface.core.get_noise_levels`.
        If ``None``, ``get_noise_levels`` uses ``seed=0``.
    job_kwargs : dict | None, default: None
        Keyword arguments forwarded to :func:`run_node_pipeline` (e.g.
        ``n_jobs``, ``chunk_duration``).
    random_slices_kwargs : dict | None, default: None
        Additional keyword arguments forwarded to the ``random_slices_kwargs``
        argument of :func:`~spikeinterface.core.get_noise_levels`.
    return_envelope : bool, default: False
        If ``True``, also return the intermediate envelope recording so that it
        can be inspected or plotted.

    Returns
    -------
    artifacts : np.ndarray
        Array with dtype ``artifact_dtype`` describing each detected artifact
        period.  Fields: ``"start_sample_index"``, ``"end_sample_index"``,
        ``"segment_index"``.
    envelope : BaseRecording
        Only returned when ``return_envelope=True``.  The processed envelope
        recording (rectified → Gaussian-filtered → common-average referenced).
    """
    envelope = RectifyRecording(recording)
    envelope = GaussianFilterRecording(envelope, freq_min=None, freq_max=freq_max)
    if apply_envelope_common_reference:
        envelope = CommonReferenceRecording(envelope)

    job_kwargs = fix_job_kwargs(job_kwargs)
    if random_slices_kwargs is None:
        random_slices_kwargs = {}
    else:
        random_slices_kwargs = random_slices_kwargs.copy()
    random_slices_kwargs["seed"] = seed
    random_data = get_random_data_chunks(envelope, **random_slices_kwargs)
    medians = np.median(random_data, axis=0)
    mad = np.median(np.abs(random_data - medians), axis=0)
    mads = mad / 0.6745

    node0 = _DetectThresholdCrossing(
        envelope,
        detect_threshold=detect_threshold,
        mads=mads,
        medians=medians,
    )

    threshold_crossings = run_node_pipeline(
        envelope,
        [node0],
        job_kwargs,
        job_name="detect artifact on  envelope",
        check_for_peak_source=False,
    )

    order = np.lexsort((threshold_crossings["sample_index"], threshold_crossings["segment_index"]))
    threshold_crossings = threshold_crossings[order]

    artifacts = _transform_internal_dtype_to_artifact_dtype(threshold_crossings, recording)

    num_samples = [recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())]
    artifacts = _collapse_events(artifacts)

    if return_envelope:
        return artifacts, envelope
    else:
        return artifacts


def _transform_internal_dtype_to_artifact_dtype(
    artifacts: np.ndarray,
    recording: BaseRecording,
) -> np.ndarray:
    """
    Convert threshold-crossing events to the standard ``artifact_dtype`` format.

    Threshold-crossing events are stored as individual rising/falling edge
    records.  This function pairs them up segment by segment to produce
    contiguous period records.  Edge cases at segment boundaries are handled:

    * If the first event in a segment is a falling edge, an implicit rising
      edge at sample 0 is prepended.
    * If the last event in a segment is a rising edge, an implicit falling edge
      at the last sample of the segment is appended.

    Parameters
    ----------
    artifacts : np.ndarray
        Array of threshold-crossing events with fields ``"sample_index"``,
        ``"segment_index"``, and ``"front"`` (``True`` = rising edge).
        Must be sorted by ``(segment_index, sample_index)``.
    recording : BaseRecording
        The original recording, used to determine the number of segments and
        the number of samples per segment.

    Returns
    -------
    np.ndarray
        Array with dtype ``artifact_dtype`` containing the merged artifact
        periods.  Returns an empty array if no crossings are found.
    """
    num_seg = recording.get_num_segments()

    final_artifacts = []
    for seg_index in range(num_seg):
        mask = artifacts["segment_index"] == seg_index
        sub_thr = artifacts[mask]
        print(sub_thr)
        if len(sub_thr) > 0:
            if not sub_thr["front"][0]:
                local_thr = np.zeros(1, dtype=np.dtype(base_period_dtype + [("front", "bool")]))
                local_thr["sample_index"] = 0
                local_thr["front"] = True
                sub_thr = np.hstack((local_thr, sub_thr))
            if sub_thr["front"][-1]:
                local_thr = np.zeros(1, dtype=np.dtype(base_period_dtype + [("front", "bool")]))
                local_thr["sample_index"] = recording.get_num_samples(seg_index)
                local_thr["front"] = False
                sub_thr = np.hstack((sub_thr, local_thr))

            local_artifact = np.zeros(int(np.ceil(sub_thr.size / 2)), dtype=artifact_dtype)
            local_artifact["start_sample_index"] = sub_thr["sample_index"][::2]
            local_artifact["end_sample_index"] = sub_thr["sample_index"][1::2]
            local_artifact["segment_index"] = seg_index
            final_artifacts.append(local_artifact)

    if len(final_artifacts) > 0:
        final_artifacts = np.concatenate(final_artifacts)
    else:
        final_artifacts = np.zeros(0, dtype=artifact_dtype)
    return final_artifacts


_method_to_function = {
    "envelope": detect_artifact_periods_by_envelope,
    "saturation": detect_saturation_periods,
}


def detect_artifact_periods(
    recording: BaseRecording,
    method: Literal["envelope", "saturation"] = "envelope",
    method_kwargs: dict | None = None,
    job_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Detect artifact periods using one of several available methods.

    Available methods:

    * ``"envelope"``: detects artifacts as threshold crossings of a low-pass-filtered, rectified
      channel envelope.
    * ``"saturation"``: detects amplifier saturation events by a voltage threshold and/or a derivative threshold.

    See the documentation of each sub-function for a full description of their
    parameters, which can be forwarded via ``method_kwargs``.

    Parameters
    ----------
    recording : BaseRecording
        The recording on which to detect artifact periods.
    method : {"envelope", "saturation"}, default: "envelope"
        Detection method to use.
    method_kwargs : dict | None, default: None
        Additional keyword arguments forwarded to the selected detection
        function.  Pass ``None`` to use that function's defaults.
    job_kwargs : dict | None, default: None
        Keyword arguments forwarded to :func:`run_node_pipeline` (e.g.
        ``n_jobs``, ``chunk_duration``).

    Returns
    -------
    np.ndarray
        Array with dtype ``artifact_dtype`` describing each detected artifact
        period.
    """
    assert (
        method in _method_to_function
    ), f"Method {method} not recognized. Valid methods are: {_method_to_function.keys()}"
    if method_kwargs is None:
        method_kwargs = dict()

    artifact_periods = _method_to_function[method](recording, job_kwargs=job_kwargs, **method_kwargs)

    return artifact_periods
