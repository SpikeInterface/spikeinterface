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


class DetectSaturation(PeakDetector):

    name = "detect_saturation"
    preferred_mp_context = None

    def __init__(
        self,
        recording,
        saturation_threshold=5,  # TODO: FIX,   max_voltage = max_voltage if max_voltage is not None else sr.range_volts[:-1]
        voltage_per_sec_threshold=5,  # TODO: completely arbitrary default value
        proportion=0.5,  # TODO: guess
        mute_window_samples=7,  # TODO: check
    ):

        # TODO: fix name
        # TODO: review this
        EVENT_VECTOR_TYPE = [
            ("start_sample_index", "int64"),
            ("stop_sample_index", "int64"),
            ("segment_index", "int64"),
            ("channel_x_start", "float64"),
            ("channel_x_stop", "float64"),
            ("channel_y_start", "float64"),
            ("channel_y_stop", "float64"),
            ("method_id", "U128"),
        ]
        self.voltage_per_sec_threshold = voltage_per_sec_threshold
        self.saturation_threshold = saturation_threshold
        self.sampling_frequency = recording.get_sampling_frequency()
        self.proportion = proportion
        self.mute_window_samples = mute_window_samples

        self._dtype = EVENT_VECTOR_TYPE

    def get_trace_margin(self):  # TODO: add margin
        return 0

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):  # TODO: required arguments
        """
        Computes
        :param data: [nc, ns]: voltage traces array
        :param max_voltage: maximum value of the voltage: scalar or array of size nc (same units as data)
        :param v_per_sec: maximum derivative of the voltage in V/s (or units/s)
        :param fs: sampling frequency Hz (defaults to 30kHz)
        :param proportion: 0 < proportion <1  of channels above threshold to consider the sample as saturated (0.2)
        :param mute_window_samples=7: number of samples for the cosine taper applied to the saturation
        :return:
            saturation [ns]: boolean array indicating the saturated samples
            mute [ns]: float array indicating the mute function to apply to the data [0-1]
        """
        import scipy  # TODO: handle import

        max_voltage = self.saturation_threshold
        v_per_sec = self.voltage_per_sec_threshold
        fs = self.sampling_frequency
        proportion = self.proportion
        mute_window_samples = self.mute_window_samples

        data = traces.T  # TODO: handle

        # first computes the saturated samples
        max_voltage = np.atleast_1d(max_voltage)[:, np.newaxis]
        saturation = np.mean(np.abs(data) > max_voltage * 0.98, axis=0)

        # then compute the derivative of the voltage saturation
        n_diff_saturated = np.mean(np.abs(np.diff(data, axis=-1)) / fs >= v_per_sec, axis=0)
        n_diff_saturated = np.r_[n_diff_saturated, 0]

        # if either of those reaches more than the proportion of channels labels the sample as saturated
        saturation = np.logical_or(saturation > proportion, n_diff_saturated > proportion)

        # apply a cosine taper to the saturation to create a mute function
        win = scipy.signal.windows.cosine(mute_window_samples)
        mute = np.maximum(0, 1 - scipy.signal.convolve(saturation, win, mode="same"))
        return saturation, mute

        # z = np.median(traces / self.abs_thresholds, 1)
        # threshold_mask = np.diff((z > 1) != 0, axis=0)
        # indices = np.flatnonzero(threshold_mask)
        # threshold_crossings = np.zeros(indices.size, dtype=self._dtype)
        # threshold_crossings["sample_index"] = indices
        # threshold_crossings["front"][::2] = True
        # threshold_crossings["front"][1::2] = False
        # return (threshold_crossings,)


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

    **noise_levels_kwargs : Keyword arguments for `spikeinterface.core.get_noise_levels()` function

    """
    from spikeinterface.core.node_pipeline import (  # TODO: ask can we import this at the top?
        run_node_pipeline,
    )

    _, job_kwargs = split_job_kwargs(noise_levels_kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    node0 = DetectSaturation(
        recording,
        seed=seed, **noise_levels_kwargs
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
