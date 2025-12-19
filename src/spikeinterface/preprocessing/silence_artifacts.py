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
        self._dtype = np.dtype(base_peak_dtype + [("front", "bool")])

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
        threshold_crossings["front"][::2] = True
        threshold_crossings["front"][1::2] = False
        return (threshold_crossings,)


def detect_period_artifacts_by_envelope(
    recording,
    detect_threshold=5,
    min_duration_ms=50,
    freq_max=20.0,
    seed=None,
    noise_levels=None,
    **noise_levels_kwargs,
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
    min_duration_ms : float, default: 50
        The minimum duration for a threshold crossing to be considered as an artefact.
    noise_levels : array
        Noise levels if already computed
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

    _, job_kwargs = split_job_kwargs(noise_levels_kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    node0 = DetectThresholdCrossing(
        recording, detect_threshold=detect_threshold, noise_levels=noise_levels, seed=seed, **noise_levels_kwargs
    )

    threshold_crossings = run_node_pipeline(
        recording,
        [node0],
        job_kwargs,
        job_name="detect threshold crossings",
    )

    order = np.lexsort((threshold_crossings["sample_index"], threshold_crossings["segment_index"]))
    threshold_crossings = threshold_crossings[order]

    periods = []
    fs = recording.sampling_frequency
    max_duration_samples = int(min_duration_ms * fs / 1000)
    num_seg = recording.get_num_segments()

    for seg_index in range(num_seg):
        sub_periods = []
        mask = threshold_crossings["segment_index"] == seg_index
        sub_thr = threshold_crossings[mask]
        if len(sub_thr) > 0:
            local_thr = np.zeros(1, dtype=np.dtype(base_peak_dtype + [("front", "bool")]))
            if not sub_thr["front"][0]:
                local_thr["sample_index"] = 0
                local_thr["front"] = True
                sub_thr = np.hstack((local_thr, sub_thr))
            if sub_thr["front"][-1]:
                local_thr["sample_index"] = recording.get_num_samples(seg_index)
                local_thr["front"] = False
                sub_thr = np.hstack((sub_thr, local_thr))

            indices = np.flatnonzero(np.diff(sub_thr["front"]))
            for i, j in zip(indices[:-1], indices[1:]):
                if sub_thr["front"][i]:
                    start = sub_thr["sample_index"][i]
                    end = sub_thr["sample_index"][j]
                    if end - start > max_duration_samples:
                        sub_periods.append((start, end))

        periods.append(sub_periods)

    return periods, envelope


class SilencedArtifactsRecording(SilencedPeriodsRecording):
    """
    Silence user-defined periods from recording extractor traces. The code will construct
    an enveloppe of the recording (as a low pass filtered version of the traces) and detect
    threshold crossings to identify the periods to silence. The periods are then silenced either
    on a per channel basis or across all channels by replacing the values by zeros or by
    adding gaussian noise with the same variance as the one in the recordings

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to silence putative artifacts
    detect_threshold : float, default: 5
        The threshold to detect artifacts. The threshold is computed as `detect_threshold * noise_level`
    freq_max : float, default: 20
        The maximum frequency for the low pass filter used
    min_duration_ms : float, default: 50
        The minimum duration for a threshold crossing to be considered as an artefact.
    noise_levels : array
        Noise levels if already computed
    seed : int | None, default: None
        Random seed for `get_noise_levels` and `NoiseGeneratorRecording`.
        If none, `get_noise_levels` uses `seed=0` and `NoiseGeneratorRecording` generates a random seed using `numpy.random.default_rng`.
    mode : "zeros" | "noise", default: "zeros"
        Determines what periods are replaced by. Can be one of the following:

        - "zeros": Artifacts are replaced by zeros.

        - "noise": The periods are filled with a gaussion noise that has the
                   same variance that the one in the recordings, on a per channel
                   basis
    **noise_levels_kwargs : Keyword arguments for `spikeinterface.core.get_noise_levels()` function

    Returns
    -------
    silenced_recording : SilencedArtifactsRecording
        The recording extractor after silencing detected artifacts
    """

    _precomputable_kwarg_names = ["list_periods"]

    def __init__(
        self,
        recording,
        detect_threshold=5,
        verbose=False,
        freq_max=20.0,
        min_duration_ms=50,
        mode="zeros",
        noise_levels=None,
        seed=None,
        list_periods=None,
        **noise_levels_kwargs,
    ):

        if list_periods is None:
            list_periods, _ = detect_period_artifacts_by_envelope(
                recording,
                detect_threshold=detect_threshold,
                min_duration_ms=min_duration_ms,
                freq_max=freq_max,
                seed=seed,
                noise_levels=noise_levels,
                **noise_levels_kwargs,
            )

            if verbose:
                for i, periods in enumerate(list_periods):
                    total_time = np.sum([end - start for start, end in periods])
                    percentage = 100 * total_time / recording.get_num_samples(i)
                    print(f"{percentage}% of segment {i} has been flagged as artifactual")

        SilencedPeriodsRecording.__init__(
            self, recording, list_periods, mode=mode, noise_levels=noise_levels, seed=seed, **noise_levels_kwargs
        )


# function for API
silence_artifacts = define_function_handling_dict_from_class(
    source_class=SilencedArtifactsRecording, name="silence_artifacts"
)
