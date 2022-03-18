import numpy as np
import scipy.signal

from .basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorSegment,
)
from .filter import fix_dtype

from scipy import signal


class ResampleRecording(BasePreprocessor):
    """
    Resample the recording extractor traces.

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
        # in case there was a time_vector, it will be dropped for sanity.
        for parent_segment in recording._recording_segments:
            parent_segment.time_vector = None
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
        parent_samp_fr = self.parent_recording_segment.sampling_frequency
        current_samp_fr = self.sampling_frequency
        # to get the start and end frames from the parent:
        # parent_frame = (frame / current_samp_fr) * parent_samp_fr
        # One could add a margin
        # beware of start_ or end_frame as none:
        if start_frame == None:
            start_frame = 0
        if end_frame == None:
            end_frame = self.get_num_samples()
        parent_start_frame, parent_end_frame = [
            int((frame / current_samp_fr) * parent_samp_fr)
            for frame in [start_frame, end_frame]
        ]
        #  get traces and resample
        parent_traces = self.parent_recording_segment.get_traces(
            parent_start_frame,
            parent_end_frame,
            channel_indices
        )
        # Check which method to use:
        if np.mod(parent_samp_fr, current_samp_fr) == 0:
            # the
            q = parent_samp_fr // current_samp_fr
            # Decimate can have issues for some cases, returning NaNs
            resampled_traces = signal.decimate(parent_traces, q=q, axis=0)
            # If that's the case, use signal.resample
            if np.any(np.isnan(resampled_traces)):
                resampled_traces = signal.resample(parent_traces, int(end_frame - start_frame), axis=0)
        else:
            resampled_traces = signal.resample(parent_traces, int(end_frame - start_frame), axis=0)
        return resampled_traces.astype(self._dtype)


# functions for API
def resample(*arg, **kwargs):
    return ResampleRecording(*arg, **kwargs)


resample.__doc__ = ResampleRecording.__doc__
