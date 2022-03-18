import numpy as np
import scipy.signal

from .basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)
from .filter import fix_dtype

try:
    from scipy import special, signal

    HAVE_RR = True
except ImportError:
    HAVE_RR = False


class ResampleRecording(BasePreprocessor):
    """
    Generic resample class based the previous SpikeToolkits API.

    If the resampling rate is multiple of the sampling rate, the faster
    scipy decimate function is used.

    Parameters
    ----------
    recording: Recording
        The recording extractor to be re-referenced
    resample_rate: float or list
        The resampling frequency
    dtype: dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used

    Returns
    -------
    resample_recording: ResampleRecording
        The resampled recording extractor object

    """

    name = "resample"

    def __init__(
        self,
        recording,
        resample_rate,
        dtype=None,
    ):
        # Original sampling frequency
        self._orig_samp_freq = recording.get_sampling_frequency()
        self._resample_rate = resample_rate
        self._sampling_frequency = resample_rate
        dtype = fix_dtype(recording, dtype)
        self.dtype = dtype

        BasePreprocessor.__init__(
            self, recording, sampling_frequency=resample_rate, dtype=dtype
        )

        for parent_segment in recording._recording_segments:
            self.add_recording_segment(
                ResampleRecordingSegment(
                    parent_segment,
                    resample_rate,
                    dtype,
                )
            )

        self._kwargs = dict(
            recording=recording.to_dict(),
            resample_rate=resample_rate,
            dtype=dtype,
        )


class ResampleRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, resample_rate, dtype):
        # After setting the preprocessor update the sampling rate
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.sampling_frequency = resample_rate
        self._resample_rate = resample_rate
        self._dtype = dtype

    def get_num_samples(self):
        return int(
            self.parent_recording_segment.get_num_samples()
            / self.parent_recording_segment.sampling_frequency
            * self._resample_rate
        )

    def get_times(self):
        return super().get_times()

    def get_traces(self, start_frame, end_frame, channel_indices):

        # parent_frame = (curr_frame_n / current_samp_rate) * parent_samp_rate
        parent_start_frame, parent_end_frame = [
            int(
                (frame / self.sampling_frequency)
                * self.parent_recording_segment.sampling_frequency
            )
            for frame in [start_frame, end_frame]
        ]

        parent_traces = self.parent_recording_segment.get_traces(
            parent_start_frame, parent_end_frame, slice(None)
        )
        resampled_traces = signal.resample(
            parent_traces, int(end_frame - start_frame), axis=0
        )
        return resampled_traces.astype(self._dtype)


# functions for API
def resample(*arg, **kwargs):
    return ResampleRecording(*arg, **kwargs)


resample.__doc__ = ResampleRecording.__doc__
