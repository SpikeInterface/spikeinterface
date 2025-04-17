from __future__ import annotations

import numpy as np
import warnings

from spikeinterface.core.core_tools import (
    define_function_handling_dict_from_class,
    recursive_key_finder,
)

from .basepreprocessor import BasePreprocessor
from .filter import fix_dtype
from spikeinterface.core import get_chunk_with_margin, BaseRecordingSegment


class ResampleRecording(BasePreprocessor):
    """
    Resample the recording extractor traces.

    If the original sampling rate is multiple of the resample_rate, it will use
    the signal.decimate method from scipy. In other cases, it uses signal.resample. In the
    later case, the resulting signal can have issues on the edges, mainly on the
    rightmost.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    resample_rate : int
        The resampling frequency
    margin_ms : float, default: 100.0
        Margin in ms for computations, will be used to decrease edge effects.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used.
    skip_checks : bool, default: False
        If True, checks on sampling frequencies and cutoff filter frequencies are skipped

    Returns
    -------
    resample_recording : ResampleRecording
        The resampled recording extractor object.

    """

    def __init__(
        self,
        recording,
        resample_rate,
        margin_ms=100.0,
        dtype=None,
        skip_checks=False,
    ):
        # Floating point resampling rates can lead to unexpected results, avoid actively
        msg = "Non integer resampling rates can lead to unexpected results."
        assert isinstance(resample_rate, (int, np.integer)), msg
        # Original sampling frequency
        self._orig_samp_freq = recording.get_sampling_frequency()
        self._resample_rate = resample_rate
        self._sampling_frequency = resample_rate
        # fix_dtype not always returns the str, make sure it does
        dtype = fix_dtype(recording, dtype).str
        # Ensure that the requested resample rate is doable:
        if skip_checks:
            assert check_nyquist(recording, resample_rate), "The requested resample rate would induce errors!"

        # Get a margin to avoid issues later
        margin = int(margin_ms * recording.get_sampling_frequency() / 1000)

        BasePreprocessor.__init__(self, recording, sampling_frequency=resample_rate, dtype=dtype)
        # in case there was a time_vector, it will be dropped for sanity.
        for parent_segment in recording._recording_segments:
            parent_segment.time_vector = None
            self.add_recording_segment(
                ResampleRecordingSegment(
                    parent_segment,
                    resample_rate,
                    recording.get_sampling_frequency(),
                    margin,
                    dtype,
                )
            )

        self._kwargs = dict(
            recording=recording,
            resample_rate=resample_rate,
            margin_ms=margin_ms,
            dtype=dtype,
            skip_checks=skip_checks,
        )


class ResampleRecordingSegment(BaseRecordingSegment):
    def __init__(
        self,
        parent_recording_segment,
        resample_rate,
        parent_rate,
        margin,
        dtype,
    ):
        # Do not use BasePreprocessorSegment bcause we have to reset the sampling rate!
        BaseRecordingSegment.__init__(
            self,
            sampling_frequency=resample_rate,
            t_start=parent_recording_segment.t_start,
        )
        self._parent_segment = parent_recording_segment
        self._parent_rate = parent_rate
        self._margin = margin
        self._dtype = dtype

    def get_num_samples(self):
        return int(self._parent_segment.get_num_samples() / self._parent_rate * self.sampling_frequency)

    def get_traces(self, start_frame, end_frame, channel_indices):
        # get parent traces with margin
        parent_start_frame, parent_end_frame = [
            int((frame / self.sampling_frequency) * self._parent_rate) for frame in [start_frame, end_frame]
        ]
        parent_traces, left_margin, right_margin = get_chunk_with_margin(
            self._parent_segment,
            parent_start_frame,
            parent_end_frame,
            channel_indices,
            self._margin,
            add_reflect_padding=True,
            dtype=np.float32,
        )
        # get left and right margins for the resampled case
        left_margin_rs, right_margin_rs = [
            int((margin / self._parent_rate) * self.sampling_frequency) for margin in [left_margin, right_margin]
        ]

        # get the size for the resampled traces in case of resample:
        num = int((end_frame + right_margin_rs) - (start_frame - left_margin_rs))

        # Decimate can misbehave on some cases, while resample always looks nice enough.
        # Check which method to use:
        from scipy import signal

        if np.mod(self._parent_rate, self.sampling_frequency) == 0:
            # Ratio between sampling frequencies
            q = int(self._parent_rate / self.sampling_frequency)
            # Decimate can have issues for some cases, returning NaNs
            resampled_traces = signal.decimate(parent_traces, q=q, axis=0)
            # If that's the case, use signal.resample
            if np.any(np.isnan(resampled_traces)):
                resampled_traces = signal.resample(parent_traces, num, axis=0)
        else:
            resampled_traces = signal.resample(parent_traces, num, axis=0)

        # now take care of the edges
        resampled_traces = resampled_traces[left_margin_rs : num - right_margin_rs]
        return resampled_traces.astype(self._dtype)


resample = define_function_handling_dict_from_class(source_class=ResampleRecording, name="resample")


# Some helpers to do checks
def check_nyquist(recording, resample_rate):
    # Check that the original and requested sampling rates will not induce aliasing
    # Basic test, compare the sampling frequency with the resample rate
    sampling_frequency_check = recording.get_sampling_frequency() / 2 > resample_rate
    # Check that the signal, if it has been filtered, is still not violating
    if recording.is_filtered():
        # Check if we have access to the highcut frequency
        freq_max = list(recursive_key_finder(recording, "freq_max"))
        if freq_max:
            # Given that there might be more than one filter applied, keep the lowest
            freq_max = min(freq_max)
            lowpass_cutoff_check = freq_max / 2 > resample_rate
        else:
            # If has been filterd but unknown high cutoff, give warning and asume the best
            warnings.warn("The recording is filtered, but we can't ensure that it complies with the Nyquist limit.")
            lowpass_cutoff_check = True
    else:
        # If it hasn't been filtered, we only depend on the previous test
        warnings.warn(
            "The recording is not filtered, so cutoff frequencies cannot be checked. " "Use resampling with caution"
        )
        lowpass_cutoff_check = True
    return all([sampling_frequency_check, lowpass_cutoff_check])
