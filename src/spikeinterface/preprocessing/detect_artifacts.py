from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from spikeinterface.preprocessing.silence_periods import SilencedPeriodsRecording
from spikeinterface.preprocessing.rectify import RectifyRecording
from spikeinterface.preprocessing.common_reference import CommonReferenceRecording
from spikeinterface.preprocessing.filter_gaussian import GaussianFilterRecording
from spikeinterface.core.job_tools import split_job_kwargs, fix_job_kwargs
from spikeinterface.core.recording_tools import get_noise_levels
from spikeinterface.core.node_pipeline import PeakDetector, base_peak_dtype
import numpy as np


artifact_dtype = [
    ("start_index", "int64"),
    ("stop_index", "int64"),
    ("segment_index", "int64"),
]

extended_artifact_dtype = artifact_dtype + [
    # TODO
]


_internal_dtype = [
    ("sample_index", "int64"),
    ("segment_index", "int64"),
    ("front", "bool")
]


def detect_artifact_periods(
    recording,
    method="envelope",
    method_kwargs=None,
    job_kwargs=None,
):
    """

    """

    if method_kwargs is None:
        method_kwargs = dict()

    if method == "envelope":
        artifacts, envelope = detect_period_artifacts_by_envelope(recording, **method_kwargs, job_kwargs=job_kwargs)
    elif method == "saturation":
        raise NotImplementedError("Soon")
    
    else:
        raise ValueError("")
    
    return artifacts



## detect_period_artifacts_saturation Zone




## detect_period_artifacts_by_envelope Zone

class DetectThresholdCrossing(PeakDetector):

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
        self._dtype = np.dtype(_internal_dtype)

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


def detect_period_artifacts_by_envelope(
    recording,
    detect_threshold=5,
    # min_duration_ms=50,
    freq_max=20.0,
    seed=None,
    job_kwargs=None,
    random_slices_kwargs=None,
):
    """
    Docstring for detect_period_artifacts. Function to detect putative artifact periods as threshold crossings of
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

    from spikeinterface.core.node_pipeline import (
        run_node_pipeline,
    )

    # _, job_kwargs = split_job_kwargs(noise_levels_kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)
    if random_slices_kwargs is None:
        random_slices_kwargs = {}
    else:
        random_slices_kwargs = random_slices_kwargs.copy()
    random_slices_kwargs["seed"] = seed
    noise_levels = get_noise_levels(envelope, return_in_uV=False, random_slices_kwargs=random_slices_kwargs)

    node0 = DetectThresholdCrossing(
        recording, detect_threshold=detect_threshold, noise_levels=noise_levels, seed=seed,
    )

    threshold_crossings = run_node_pipeline(
        envelope,
        [node0],
        job_kwargs,
        job_name="detect threshold crossings",
    )

    order = np.lexsort((threshold_crossings["sample_index"], threshold_crossings["segment_index"]))
    threshold_crossings = threshold_crossings[order]

    artifacts = _transform_internal_dtype_to_artifact_dtype(threshold_crossings, recording)

    
    return artifacts, envelope


# tools

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
            
            local_artifact = np.zeros(sub_thr.size/2, dtype=artifact_dtype)
            local_artifact["start_index"] = sub_thr["sample_index"][::2]
            local_artifact["stop_index"] = sub_thr["sample_index"][1::2]
            local_artifact["segment_index"] = seg_index
            final_artifacts.append(local_artifact)
    
    if len(final_artifacts) > 0:
        final_artifacts = np.concatenate(final_artifacts)
    else:
        final_artifacts = np.zeros(0, dtype=artifact_dtype)
    return final_artifacts