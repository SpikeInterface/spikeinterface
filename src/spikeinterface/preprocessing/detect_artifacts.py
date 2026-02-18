from __future__ import annotations

import numpy as np

from spikeinterface.core.base import base_period_dtype

# from spikeinterface.core.core_tools import define_function_handling_dict_from_class
# from spikeinterface.preprocessing.silence_periods import SilencedPeriodsRecording
from spikeinterface.preprocessing.rectify import RectifyRecording
from spikeinterface.preprocessing.common_reference import CommonReferenceRecording
from spikeinterface.preprocessing.filter_gaussian import GaussianFilterRecording
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.recording_tools import get_noise_levels
from spikeinterface.core.node_pipeline import PeakDetector, base_peak_dtype, run_node_pipeline, PipelineNode
import numpy as np

artifact_dtype = base_period_dtype


# this will be extend with channel boundaries if needed
# extended_artifact_dtype = artifact_dtype + [
#     # TODO
# ]


def detect_artifact_periods(
    recording,
    method="envelope",
    method_kwargs=None,
    job_kwargs=None,
):
    """
    Detect artifacts with several possible methods:
      * 'saturation' using detect_artifact_periods_by_envelope()
      * 'envelope' using detect_saturation_periods()

    See sub methods for more information on parameters.
    """

    if method_kwargs is None:
        method_kwargs = dict()

    if method == "envelope":
        artifact_periods, envelope = detect_artifact_periods_by_envelope(
            recording, **method_kwargs, job_kwargs=job_kwargs
        )
    elif method == "saturation":
        artifact_periods = detect_saturation_periods(recording, **method_kwargs, job_kwargs=job_kwargs)
    else:
        raise ValueError(f"detect_artifact_periods() method='{method}' is not valid")

    return artifact_periods


## detect_period_artifacts_saturation Zone


def _collapse_events(events):
    """
    If events are detected at a chunk edge, they will be split in two.
    This detects such cases and collapses them in a single record instead.
    """
    order = np.lexsort((events["start_sample_index"], events["segment_index"]))
    events = events[order]
    to_drop = np.zeros(events.size, dtype=bool)

    # compute if duplicate
    for i in np.arange(events.size - 1):
        same = events["end_sample_index"][i] == events["start_sample_index"][i + 1]
        if same:
            to_drop[i] = True
            events["start_sample_index"][i + 1] = events["start_sample_index"][i]

    return events[~to_drop].copy()


class _DetectSaturation(PipelineNode):
    """
    A recording node for parallelising saturation detection.

    Run with `run_node_pipeline`, this computes saturation events
    for a given chunk. See `detect_saturation()` for details.
    """

    name = "detect_saturation"
    preferred_mp_context = None
    _compute_has_extended_signature = True

    def __init__(
        self,
        recording,
        saturation_threshold_uV,
        uV_per_ms_threshold,
        proportion,
    ):
        PipelineNode.__init__(self, recording, return_output=True)

        num_chans = recording.get_num_channels()

        self.uV_per_ms_threshold = uV_per_ms_threshold
        thresh = np.full((num_chans,), saturation_threshold_uV)
        # 0.98 is empirically determined as the true saturating point is
        # slightly lower than the documented saturation point of the probe
        self.sampling_frequency = recording.get_sampling_frequency()
        self.proportion = proportion
        self._dtype = np.dtype(artifact_dtype)
        self.gain = recording.get_channel_gains()
        self.offset = recording.get_channel_offsets()

        self.saturation_threshold_unscaled = (thresh - self.offset) / self.gain * 0.98

        # do not apply offset when dealing with the derivative
        self.uV_per_ms_threshold = (uV_per_ms_threshold * self.sampling_frequency / 1e3) / self.gain

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        """
        Compute saturation events for a given chunk of data.
        See `detect_saturation()` for details.
        """
        saturation = np.mean(np.abs(traces) > self.saturation_threshold_unscaled, axis=1)

        if self.uV_per_ms_threshold is not None:
            fs = self.sampling_frequency
            # then compute the derivative of the voltage saturation

            n_diff_saturated = np.mean(np.abs(np.diff(traces, axis=0)) >= self.uV_per_ms_threshold, axis=1)

            # Note this means the velocity is not checked for the last sample in the
            # check because we are taking the forward derivative
            n_diff_saturated = np.r_[n_diff_saturated, 0]

            # if either of those reaches more than the proportion of channels labels the sample as saturated
            saturation = np.logical_or(saturation > self.proportion, n_diff_saturated > self.proportion)
        else:
            saturation = saturation > self.proportion

        intervals = np.flatnonzero(np.diff(saturation, prepend=False, append=False))
        n_events = len(intervals) // 2  # Number of saturation periods
        events = np.zeros(n_events, dtype=artifact_dtype)

        for i, (start, stop) in enumerate(zip(intervals[::2], intervals[1::2])):
            events[i]["start_sample_index"] = start + start_frame
            events[i]["end_sample_index"] = stop + start_frame
            events[i]["segment_index"] = segment_index

        return (events,)


def detect_saturation_periods(
    recording,
    saturation_threshold_uV,  # 1200 uV
    uV_per_ms_threshold=None,  # 1e-8 V.s-1
    proportion=0.2,
    job_kwargs=None,
):
    """
        Detect amplifier saturation events (either single sample or multi-sample periods) in the data.
        Saturation detection with this function should be applied to the raw data, before preprocessing.
        However, saturation periods detected should be zeroed out after preprocessing has been performed.

        Saturation is detected by a voltage threshold, and optionally a derivative threshold that
        flags periods of high velocity changes in the voltage. See _DetectSaturation.compute()
        for details on the algorithm.

        Parameters
        ----------
        recording : BaseRecording
            The recording on which to detect the saturation events.
        saturation_threshold_uV : float
            The voltage saturation threshold in volts. This will depend on the recording
            probe and amplifier gain settings. For NP1 the value of 1200 uV is recommended (IBL).
            Note that NP2 probes are more difficult to saturate than NP1.
        uV_per_ms_threshold : None | float
            The first-derivative threshold in volts per second. Periods of the data over which the change
            in velocity is greater than this threshold will be detected as saturation events. Use `None` to
            skip this method and only use `saturation_threshold_uV` for detection. Otherwise, the value should be
            empirically determined (IBL use 1e-8 V.s-1) for NP1 probes.

        proportion : float
            0 < proportion <1  of channels above threshold to consider the sample as saturated
        mute_window_samples : int
            TODO: should we scale this based on the fs?
        job_kwargs: dict
            The classical job_kwargs

        most useful for NP1
        can use ratio as a intuition for the value but dont do it in code

        Returns
    -------
        collapsed_events : np.recarray
            A numpy recarray holding information on each saturation event. Has the fields:
            "start_sample_index", "stop_sample_index", "segment_index", "method_id"
    """
    if job_kwargs:
        job_kwargs = {}

    job_kwargs = fix_job_kwargs(job_kwargs)

    node0 = _DetectSaturation(
        recording,
        saturation_threshold_uV=saturation_threshold_uV,
        uV_per_ms_threshold=uV_per_ms_threshold,
        proportion=proportion,
    )

    saturation_periods = run_node_pipeline(
        recording, [node0], job_kwargs=job_kwargs, job_name="detect saturation artifacts"
    )

    return _collapse_events(saturation_periods)


## detect_artifact_periods_by_envelope Zone


class _DetectThresholdCrossing(PeakDetector):

    name = "threshold_crossings"
    preferred_mp_context = None

    def __init__(
        self,
        recording,
        detect_threshold=5,
        noise_levels=None,
        seed=None,
        noise_levels_kwargs=dict(),
    ):
        PeakDetector.__init__(self, recording, return_output=True)
        if noise_levels is None:
            random_slices_kwargs = noise_levels_kwargs.pop("random_slices_kwargs", {}).copy()
            random_slices_kwargs["seed"] = seed
            noise_levels = get_noise_levels(recording, return_in_uV=False, random_slices_kwargs=random_slices_kwargs)
        self.abs_thresholds = noise_levels * detect_threshold
        # internal dtype
        self._dtype = np.dtype([("sample_index", "int64"), ("segment_index", "int64"), ("front", "bool")])

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        z = np.median(traces / self.abs_thresholds, 1)
        threshold_mask = np.diff((z > 1) != 0, axis=0)
        indices = np.flatnonzero(threshold_mask)
        threshold_crossings = np.zeros(indices.size, dtype=self._dtype)
        threshold_crossings["sample_index"] = indices
        threshold_crossings["segment_index"] = segment_index
        threshold_crossings["front"][::2] = True
        threshold_crossings["front"][1::2] = False
        return (threshold_crossings,)


def detect_artifact_periods_by_envelope(
    recording,
    detect_threshold=5,
    # min_duration_ms=50,
    freq_max=20.0,
    seed=None,
    job_kwargs=None,
    random_slices_kwargs=None,
):
    """
    Function to detect putative artifact periods as threshold crossings of
    a global envelope of the channels.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to detect putative artifacts
    detect_threshold : float, default: 5
        The threshold to detect artifacts. The threshold is computed as `detect_threshold * noise_level`
    freq_max : float, default: 20
        The maximum frequency for the low pass filter used
    seed : int | None, default: None
        Random seed for `get_noise_levels`.
        If none, `get_noise_levels` uses `seed=0`.
    **noise_levels_kwargs : Keyword arguments for `spikeinterface.core.get_noise_levels()` function

    """

    envelope = RectifyRecording(recording)
    envelope = GaussianFilterRecording(envelope, freq_min=None, freq_max=freq_max)
    envelope = CommonReferenceRecording(envelope)

    job_kwargs = fix_job_kwargs(job_kwargs)
    if random_slices_kwargs is None:
        random_slices_kwargs = {}
    else:
        random_slices_kwargs = random_slices_kwargs.copy()
    random_slices_kwargs["seed"] = seed
    noise_levels = get_noise_levels(envelope, return_in_uV=False, random_slices_kwargs=random_slices_kwargs)

    node0 = _DetectThresholdCrossing(
        envelope,
        detect_threshold=detect_threshold,
        noise_levels=noise_levels,
        seed=seed,
    )

    threshold_crossings = run_node_pipeline(
        envelope,
        [node0],
        job_kwargs,
        job_name="detect artifact on  envelope",
    )

    order = np.lexsort((threshold_crossings["sample_index"], threshold_crossings["segment_index"]))
    threshold_crossings = threshold_crossings[order]

    artifacts = _transform_internal_dtype_to_artifact_dtype(threshold_crossings, recording)

    return artifacts, envelope


def _transform_internal_dtype_to_artifact_dtype(artifacts, recording):

    num_seg = recording.get_num_segments()

    final_artifacts = []
    for seg_index in range(num_seg):
        mask = artifacts["segment_index"] == seg_index
        sub_thr = artifacts[mask]
        if len(sub_thr) > 0:
            if not sub_thr["front"][0]:
                local_thr = np.zeros(1, dtype=np.dtype(base_peak_dtype + [("front", "bool")]))
                local_thr["sample_index"] = 0
                local_thr["front"] = True
                sub_thr = np.hstack((local_thr, sub_thr))
            if sub_thr["front"][-1]:
                local_thr = np.zeros(1, dtype=np.dtype(base_peak_dtype + [("front", "bool")]))
                local_thr["sample_index"] = recording.get_num_samples(seg_index)
                local_thr["front"] = False
                sub_thr = np.hstack((sub_thr, local_thr))

            local_artifact = np.zeros(sub_thr.size / 2, dtype=artifact_dtype)
            local_artifact["start_index"] = sub_thr["sample_index"][::2]
            local_artifact["stop_index"] = sub_thr["sample_index"][1::2]
            local_artifact["segment_index"] = seg_index
            final_artifacts.append(local_artifact)

    if len(final_artifacts) > 0:
        final_artifacts = np.concatenate(final_artifacts)
    else:
        final_artifacts = np.zeros(0, dtype=artifact_dtype)
    return final_artifacts
