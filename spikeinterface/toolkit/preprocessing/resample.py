import numpy as np
from scipy import signal

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from .filter import fix_dtype
from .tools import get_chunk_with_margin



class ResampleRecording(BasePreprocessor):
    """
    Resample the recording extractor traces.

    If the resample_rate is multiple of the original sampling rate, the faster
    scipy.signal.decimate is used.

    Parameters
    ----------
    recording: Recording
        The recording extractor to be re-referenced
    resample_rate: int
        The resampling frequency
    margin : float (default 100)
        Margin in ms for computations, will be used to decrease edge effects.
    dtype: dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used.

    Returns
    -------
    resample_recording: ResampleRecording
        The resampled recording extractor object.

    """

    name = "resample"

    def __init__(
        self,
        recording,
        resample_rate,
        margin_ms=100.,
        dtype=None,
    ):
        # Floating point resampling rates can lead to unexpected results, avoid actively
        msg = "Non integer resampling rates can lead to unexpected results."
        assert isinstance(resample_rate, (int, np.int16, np.int32, np.int64, np.int_)), msg
        # Original sampling frequency
        self._orig_samp_freq = recording.get_sampling_frequency()
        self._resample_rate = resample_rate
        self._sampling_frequency = resample_rate
        # fix_dtype not always returns the str, make sure it does
        dtype = fix_dtype(recording, dtype).str
        self.dtype = dtype
        # Ensure that the requested resample rate is doable:
        assert check_niquist(recording, resample_rate), \
            "The requested resample rate would induce error, can't comply."

        # Get a margin to avoid issues later
        margin = int(margin_ms * recording.get_sampling_frequency() / 1000)

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
                    margin,
                    dtype,
                )
            )

        self._kwargs = dict(
            recording=recording.to_dict(),
            resample_rate=resample_rate,
            margin_ms=margin_ms,
            dtype=dtype,
        )


class ResampleRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, resample_rate, margin, dtype):
        # After setting the preprocessor update the sampling rate
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.sampling_frequency = resample_rate
        self._resample_rate = resample_rate
        self._margin = margin
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
        #  get traces with margin and resample
        parent_traces, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            parent_start_frame, parent_end_frame, channel_indices, self._margin,
            window_on_margin=False, add_zeros=False, dtype=np.float32,
        )
        # Decimate can missbehave on some cases, while resample allways looks
        # nice enough.
        # Check which method to use:
        if np.mod(parent_samp_fr, current_samp_fr) == 0:
            # Ratio between sampling frequencies
            q = int(parent_samp_fr / current_samp_fr)
            # Decimate can have issues for some cases, returning NaNs
            resampled_traces = signal.decimate(parent_traces, q=q, axis=0)
            # If that's the case, use signal.resample
            if np.any(np.isnan(resampled_traces)):
                resampled_traces = signal.resample(parent_traces, int(end_frame - start_frame), axis=0)
        else:
            resampled_traces = signal.resample(parent_traces, int(end_frame - start_frame), axis=0)
        # resampled_traces = signal.resample(parent_traces, int(end_frame - start_frame), axis=0)
        # Get left and right margins for the reampled case
        left_margin_rs, right_margin_rs = [int((margin/parent_samp_fr)*current_samp_fr) for margin in [left_margin, right_margin]]
        # Now take care of the edges:
        if right_margin > 0:
            resampled_traces = resampled_traces[left_margin_rs:-right_margin_rs, :]
        else:
            resampled_traces = resampled_traces[left_margin_rs:, :]
        return resampled_traces.astype(self._dtype)


# functions for API
def resample(*arg, **kwargs):
    return ResampleRecording(*arg, **kwargs)


resample.__doc__ = ResampleRecording.__doc__


# Some helpers to do checks
def check_niquist(recording, resample_rate):
    # Check that the original and requested sampling rates will not induce aliasing
    # Basic test, compare the sampling frequency with the resample rate
    val_rec_sf = recording.get_sampling_frequency() / 2 > resample_rate
    # Check that the signal, if it has been filtered, is still not violating
    if recording.is_filtered():
        # Check if we have access to the highcut frequency
        freq_max = list(get_nested_dict_val(recording.to_dict(), "freq_max"))
        if freq_max:
            # Given that there might be more than one filter applied, keep the lowest
            freq_max = min(freq_max)
            val_filt_mf = freq_max / 2 > resample_rate
        else:
            # If has been filterd but unknown high cutoff, give warning and asume the best
            Warning("the recording is filtered, but we can't ensure that complies with Niquist limit.")
            val_filt_mf = True
    else:
        # If it hasn't been filtered, we only depend on the previous test
        val_filt_mf = True
    return all([val_rec_sf, val_filt_mf])


def get_nested_dict_val(d, key):
    # Find all values for a key on a dictionary, even if nested
    for k, v in d.items():
        if isinstance(v, dict):
            yield from get_nested_dict_val(v, key)
        else:
            if k == key:
                yield v
