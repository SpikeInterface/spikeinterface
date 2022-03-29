import numpy as np
from scipy.fft import rfft, irfft

from .tools import get_chunk_with_margin

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment



class PhaseShiftRecording(BasePreprocessor):
    """
    This apply a phase shift to a recording to cancel the small sampling
    delay across for some recording system.
    
    This is particularly relevant for neuropixel recording.
    
    This code is from  IBL lib.
    https://github.com/int-brain-lab/ibllib/blob/master/ibllib/dsp/fourier.py
    
    Parameters
    ----------
    recording: Recording
        The recording. It need to have  "inter_sample_shift" in properties.
    margin_ms: float (default 30ms)
        margin in ms for computation
    inter_sample_shift: None or numpy array
        If "inter_sample_shift" is not in recording.properties
        we can externaly provide one.
    Returns
    -------
    filter_recording: PhaseShiftRecording
        The phase shifted recording object
    """
    name = 'phase_shift'
    def __init__(self, recording, margin_ms=30.,  inter_sample_shift=None, dtype=None):
        if inter_sample_shift is None:
            assert "inter_sample_shift" in recording.get_property_keys(), "'inter_sample_shift' is not a property!"
            sample_shifts = recording.get_property("inter_sample_shift")
        else:
            assert len(inter_sample_shift) == recording.get_num_channels(), "sample "
            sample_shifts = np.asarray(inter_sample_shift)
        
        margin = int(margin_ms * recording.get_sampling_frequency() / 1000.)


        BasePreprocessor.__init__(self, recording, dtype=dtype)
        for parent_segment in recording._recording_segments:
            rec_segment = DestripeRecordingSegment(parent_segment, sample_shifts, margin, dtype)
            self.add_recording_segment(rec_segment)
        
        # for dumpability
        if inter_sample_shift is not None:
            inter_sample_shift = list(inter_sample_shift)
        self._kwargs = dict(recording=recording.to_dict(), margin_ms=float(margin_ms),
                             inter_sample_shift=inter_sample_shift)


class DestripeRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, sample_shifts, margin, dtype):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.sample_shifts = sample_shifts
        self.margin = margin
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        traces_chunk, left_margin, right_margin = get_chunk_with_margin(self.parent_recording_segment,
                                                                        start_frame, end_frame, channel_indices,
                                                                        self.margin, add_zeros=True)
        
        need_dtype = None
        if self.dtype is not None:
            # this imply a copy always
            traces_chunk = traces_chunk.astype(self.dtype)
        else:
            if traces_chunk.dtype.kind == 'i':
                # force float computing ebcause of the taper
                need_dtype = traces_chunk.dtype
                traces_chunk = traces_chunk.astype('float32')
            else:
                # a copy is needed  because the taper is inplace
                traces_chunk = traces_chunk.copy()

        # apply some windows on border : !!!this is inplace!!!!
        taper = (1 - np.cos(np.arange(self.margin) / self.margin * np.pi)) / 2
        taper = taper[:, np.newaxis]
        traces_chunk[:self.margin] *= taper
        traces_chunk[-self.margin:] *= taper[::-1]

        traces_shift = apply_fshift_sam(traces_chunk, self.sample_shifts, axis=0)
        # traces_shift = apply_fshift_ibl(traces_chunk, self.sample_shifts, axis=0)


        traces_shift = traces_shift[left_margin:-right_margin, :]
        if need_dtype:
            traces_shift =traces_shift.astype(need_dtype)
        
        return traces_shift
        
        # TODO handle the dtype
        # return filtered_traces.astype(self.dtype)


# function for API
def phase_shift(*args, **kwargs):
    return PhaseShiftRecording(*args, **kwargs)


phase_shift.__doc__ = PhaseShiftRecording.__doc__


def apply_fshift_sam(sig, sample_shifts, axis=0):
    """
    Apply the shift on a traces buffer.
    """
    n = sig.shape[axis]
    sig_f = np.fft.rfft(sig, axis=axis)
    if n % 2 == 0:
        # n is even sig_f[-1] is nyquist and so pi
        omega = np.linspace(0, np.pi, sig_f.shape[axis])
    else:
        # n is odd sig_f[-1] is exactly nyquist!! we need (n-1) / n factor!!
        omega = np.linspace(0, np.pi * (n - 1) / n, sig_f.shape[axis])
    # broadcast omega and sample_shifts depend the axis
    if axis == 0:
        shifts = omega[:, np.newaxis] * sample_shifts[np.newaxis, :]
    else:
        shifts = omega[np.newaxis, :] * sample_shifts[:, np.newaxis]
    sig_shift = np.fft.irfft(sig_f * np.exp(- 1j  * shifts), n=n, axis=axis)
    return sig_shift

apply_fshift = apply_fshift_sam
    

def apply_fshift_ibl(w, s, axis=0, ns=None):
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
    # np.put(dephas, 1, 1)
    dephas[1] = 1
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
