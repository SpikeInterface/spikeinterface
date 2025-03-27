from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from spikeinterface.preprocessing.silence_periods import SilencedPeriodsRecording
from spikeinterface.preprocessing.rectify import RectifyRecording
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
        random_chunk_kwargs={},
    ):
        PeakDetector.__init__(self, recording, return_output=True)
        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)
        self.abs_thresholds = noise_levels * detect_threshold
        self._dtype = np.dtype(base_peak_dtype + [("onset", "bool")])

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        z = np.median(traces - self.abs_thresholds, 1)
        threshold_mask = np.diff((z > 0) != 0, axis=0)
        indices = np.flatnonzero(threshold_mask)
        local_peaks = np.zeros(indices.size, dtype=self._dtype)
        local_peaks["sample_index"] = indices
        local_peaks["onset"][::2] = True
        local_peaks["onset"][1::2] = False
        return (local_peaks,)


def detect_onsets(recording, detect_threshold=5, min_duration_ms=50, **extra_kwargs):

    from spikeinterface.core.node_pipeline import (
        run_node_pipeline,
    )

    random_chunk_kwargs, job_kwargs = split_job_kwargs(extra_kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    node0 = DetectThresholdCrossing(recording, detect_threshold, **random_chunk_kwargs)

    peaks = run_node_pipeline(
        recording,
        [node0],
        job_kwargs,
        job_name="detect threshold crossings",
    )

    periods = []
    fs = recording.sampling_frequency
    max_duration_samples = int(min_duration_ms*fs/1000)
    num_seg = recording.get_num_segments()
    for seg_index in range(num_seg):
        sub_periods = []
        mask = peaks["segment_index"] == 0
        sub_peaks = peaks[mask]
        onsets = sub_peaks[sub_peaks["onset"]]
        offsets = sub_peaks[~sub_peaks["onset"]]

        onset_time = 0
        offset_time = recording.get_num_samples(seg_index)

        while onset_time < offset_time:
            pass
        
        if len(onsets) == 0 and len(offsets) == 0:
            periods.append([])
            continue
        elif len(onsets) > 0 and len(offsets) == 0:
            offset = recording.get_num_samples(seg_index)
            if (offset - onsets["sample_index"][0]) > max_duration_samples:
                periods.append([( onsets["sample_index"][0], offset)])
            continue

        max_size = min(len(onsets), len(offsets))
        np.where(onsets[:max_size])

        for i in range(min(len(onsets), len(offsets))):
            sub_periods += [(onsets["sample_index"][i], offsets["sample_index"][i])]

        periods.append(sub_periods)
    #print(periods)
    return periods


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
    **random_chunk_kwargs : Keyword arguments for `spikeinterface.core.get_random_data_chunk()` function

    Returns
    -------
    silenced_recording : SilencedArtifactsRecording
        The recording extractor after silencing detected artifacts
    """

    def __init__(
        self,
        recording,
        detect_threshold=5,
        verbose=False,
        freq_max=5.0,
        min_duration_ms=50,
        mode="zeros",
        noise_levels=None,
        seed=None,
        list_periods=None,
        **random_chunk_kwargs,
    ):

        self.enveloppe = RectifyRecording(recording)
        self.enveloppe = GaussianFilterRecording(self.enveloppe, freq_min=None, freq_max=freq_max)

        if list_periods is None:
            list_periods = detect_onsets(self.enveloppe, 
                                         detect_threshold=detect_threshold, 
                                         min_duration_ms=min_duration_ms,
                                         **random_chunk_kwargs)
            if verbose:
                for i, periods in enumerate(list_periods):
                    total_time = np.sum([end - start for start, end in periods])
                    percentage = 100 * total_time / recording.get_num_samples(i)
                    print(f"{percentage}% of segment {i} has been flagged as artifactual")

        if "enveloppe" in random_chunk_kwargs:
            random_chunk_kwargs.pop("enveloppe")

        SilencedPeriodsRecording.__init__(
            self, recording, list_periods, mode=mode, noise_levels=noise_levels, seed=seed, **random_chunk_kwargs
        )

        self._kwargs.update(
            {
                "detect_threshold": detect_threshold,
                "freq_max": freq_max,
                "verbose": verbose,
                "min_duration_ms": min_duration_ms,
                "enveloppe": self.enveloppe,
            }
        )


# function for API
silence_artifacts = define_function_handling_dict_from_class(source_class=SilencedArtifactsRecording, name="silence_artifacts")
