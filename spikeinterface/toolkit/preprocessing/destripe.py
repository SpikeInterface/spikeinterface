import numpy as np
from scipy.fft import rfft, irfft

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class DestripeRecording(BasePreprocessor):
    name = 'destripe'

    def __init__(self, recording, inter_sample_shift=None):
        if inter_sample_shift is None:
            assert "inter_sample_shift" in recording.get_property_keys(), "'inter_sample_shift' is not a property!"
            inter_sample_shift = recording.get_property("sample_shift")
        else:
            assert len(inter_sample_shift) == recording.get_num_channels(), "..."
        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            rec_segment = DestripeRecordingSegment(parent_segment)
            self.add_recording_segment(rec_segment, inter_sample_shift)
        self._kwargs = dict(recording=recording.to_dict())


class DestripeRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, sample_shifts):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.sample_shifts = sample_shifts

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces_shift = _fshift(traces, self.sample_shifts, axis=1)
        return traces_shift


# function for API
def destripe(*args, **kwargs):
    return DestripeRecording(*args, **kwargs)


destripe.__doc__ = DestripeRecording.__doc__


def _fshift(w, s, axis=-1, ns=None):
    """
    Function from IBLIB: https://github.com/int-brain-lab/ibllib/blob/master/ibllib/dsp/fourier.py

    Shifts a 1D or 2D signal in frequency domain, to allow for accurate non-integer shifts
    :param w: input signal (if complex, need to provide ns too)
    :param s: shift in samples, positive shifts forward
    :param axis: axis along which to shift (last axis by default)
    :param axis: axis along which to shift (last axis by default)
    :param ns: if a rfft frequency domain array is provided, give a number of samples as there
     is an ambiguity
    :return: w
    """
    # create a vector that contains a 1 sample shift on the axis
    ns = ns or w.shape[axis]
    shape = np.array(w.shape) * 0 + 1
    shape[axis] = ns
    dephas = np.zeros(shape)
    np.put(dephas, 1, 1)
    dephas = rfft(dephas, axis=axis)
    # fft the data along the axis and the dephas
    do_fft = np.invert(np.iscomplexobj(w))
    if do_fft:
        W = rfft(w, axis=axis)
    else:
        W = w
    # if multiple shifts, broadcast along the other dimensions, otherwise keep a single vector
    if not np.isscalar(s):
        s_shape = np.array(w.shape)
        s_shape[axis] = 1
        s = s.reshape(s_shape)
    # apply the shift (s) to the fft angle to get the phase shift and broadcast
    W *= np.exp(1j * np.angle(dephas) * s)
    if do_fft:
        W = np.real(irfft(W, ns, axis=axis))
        W = W.astype(w.dtype)
    return W
