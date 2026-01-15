from __future__ import annotations

import numpy as np

from spikeinterface.preprocessing.silence_periods import SilencedPeriodsRecording
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.node_pipeline import run_node_pipeline
from spikeinterface.core.node_pipeline import PeakDetector


EVENT_VECTOR_TYPE = [
    ("start_sample_index", "int64"),
    ("stop_sample_index", "int64"),
    ("segment_index", "int64"),
    ("method_id", "U128"),
]


def _collapse_events(events):
    """
    If events are detected at a chunk edge, they will be split in two.
    This detects such cases and collapses them in a single record instead
    :param events:
    :return:
    """
    order = np.lexsort((events["start_sample_index"], events["segment_index"]))
    events = events[order]
    to_drop = np.zeros(events.size, dtype=bool)

    # compute if duplicate
    for i in np.arange(events.size - 1):
        same = events["stop_sample_index"][i] == events["start_sample_index"][i + 1]
        if same:
            to_drop[i] = True
            events["start_sample_index"][i + 1] = events["start_sample_index"][i]

    return events[~to_drop].copy()


class _DetectSaturation(PeakDetector):

    name = "detect_saturation"
    preferred_mp_context = None

    def __init__(
        self,
        recording,
        saturation_threshold,  # 1200 uV
        voltage_per_sec_threshold,  # 1e-8 V.s-1
        proportion,
        mute_window_samples,
    ):
        PeakDetector.__init__(self, recording, return_output=True)

        self.gains = recording.get_channel_gains()
        self.offsets = recording.get_channel_offsets()

        self.voltage_per_sec_threshold = voltage_per_sec_threshold
        self.saturation_threshold = saturation_threshold
        self.sampling_frequency = recording.get_sampling_frequency()
        self.proportion = proportion
        self.mute_window_samples = mute_window_samples
        self._dtype = EVENT_VECTOR_TYPE

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
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
        # TODO: in theory someone could have stored the scaled, float data.
        #  Document that this will scale the data!

        breakpoint()
        if traces.dtype != np.float64:
            traces = traces.astype(np.float64)  # TODO: CHECK

        traces = traces * self.gains + self.offsets

        fs = self.sampling_frequency

        # first computes the saturated samples
        max_voltage = np.atleast_1d(self.saturation_threshold)[:, np.newaxis]

        # 0.98 is empirically determined as the true saturating point is
        # slightly lower than the documented saturation point of the probe
        saturation = np.mean(np.abs(traces) > max_voltage * 0.98, axis=1)

        if self.voltage_per_sec_threshold is not None:
            # then compute the derivative of the voltage saturation
            n_diff_saturated = np.mean(np.abs(np.diff(traces, axis=0)) / fs >= self.voltage_per_sec_threshold, axis=1)
            # Note this means the velocity is not checked for the last sample in the
            # check because we are taking the forward derivative
            n_diff_saturated = np.r_[n_diff_saturated, 0]

            # if either of those reaches more than the proportion of channels labels the sample as saturated
            saturation = np.logical_or(saturation > self.proportion, n_diff_saturated > self.proportion)
        else:
            saturation = saturation > self.proportion

        intervals = np.where(np.diff(saturation, prepend=False, append=False))[0]
        n_events = len(intervals) // 2  # Number of saturation periods
        events = np.zeros(n_events, dtype=EVENT_VECTOR_TYPE)

        for i, (start, stop) in enumerate(zip(intervals[::2], intervals[1::2])):
            events[i]["start_sample_index"] = start + start_frame
            events[i]["stop_sample_index"] = stop + start_frame
            events[i]["segment_index"] = segment_index
            events[i]["method_id"] = "saturation_detection"

        # Because we inherit PeakDetector, we must expose this "sample_index"
        # array. However, it is not used and changing the value has no effect.
        toto = np.array([0], dtype=[("sample_index", "int64")])

        return (toto, events)


def detect_saturation(
    recording,
    saturation_threshold,  # 1200 uV
    voltage_per_sec_threshold,  # 1e-8 V.s-1
    proportion=0.5,
    mute_window_samples=7,
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
    saturation_threshold : float
        The voltage saturation threshold in volts. This will depend on the recording
        probe and amplifier gain settings. For NP1 the value of 1200 * 1e-6 is recommended (IBL).
        Note that NP2 probes are more difficult to saturate than NP1.
    voltage_per_sec_threshold : None | float
        The first-derivative threshold in volts per second. Periods of the data over which the change
        in velocity is greater than this threshold will be detected as saturation events. Use `None` to
        skip this method and only use `saturation_threshold` for detection. Otherwise, the value should be
        empirically determined (IBL use 1e-8 V.s-1) for NP1 probes.

    proportion :
    mute_window_samples :
    job_kwargs :

    most useful for NP1
    can use ratio as a intuition for the value but dont do it in code

    Returns
    -------

    """
    if job_kwargs:
        job_kwargs = {}

    # if saturation_threshold < 0.1:
    #    raise ValueError(f"The `saturation_threshold` should be in microvolts. "
    #                     f"Your value: {saturation_threshold} is almost certainly in volts.")

    job_kwargs = fix_job_kwargs(job_kwargs)

    node0 = _DetectSaturation(
        recording,
        saturation_threshold=saturation_threshold,
        voltage_per_sec_threshold=voltage_per_sec_threshold,
        proportion=proportion,
        mute_window_samples=mute_window_samples,
    )

    _, events = run_node_pipeline(recording, [node0], job_kwargs=job_kwargs, job_name="detect saturation events")

    return _collapse_events(events)
