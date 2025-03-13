from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class

from spikeinterface.core import get_chunk_with_margin

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class PhaseShiftRecording(BasePreprocessor):
    """
    This apply a phase shift to a recording to cancel the small sampling
    delay across for some recording system.

    This is particularly relevant for neuropixel recording.

    This code is inspired from from  IBL lib.
    https://github.com/int-brain-lab/ibllib/blob/master/ibllib/dsp/fourier.py
    and also the one from spikeglx
    https://billkarsh.github.io/SpikeGLX/help/dmx_vs_gbl/dmx_vs_gbl/

    Parameters
    ----------
    recording : Recording
        The recording. It need to have  "inter_sample_shift" in properties.
    margin_ms : float, default: 40.0
        Margin in ms for computation.
        40ms ensure a very small error when doing chunk processing
    inter_sample_shift : None or numpy array, default: None
        If "inter_sample_shift" is not in recording properties,
        we can externally provide one.
    dtype : None | str | dtype, default: None
        Dtype of input and output `recording` objects.


    Returns
    -------
    filter_recording : PhaseShiftRecording
        The phase shifted recording object
    """

    def __init__(self, recording, margin_ms=40.0, inter_sample_shift=None, dtype=None):
        if inter_sample_shift is None:
            assert "inter_sample_shift" in recording.get_property_keys(), "'inter_sample_shift' is not a property!"
            sample_shifts = recording.get_property("inter_sample_shift")
        else:
            assert (
                len(inter_sample_shift) == recording.get_num_channels()
            ), "the 'inter_sample_shift' must be same size at the num_channels "
            sample_shifts = np.asarray(inter_sample_shift)

        margin = int(margin_ms * recording.get_sampling_frequency() / 1000.0)

        if dtype is None:
            dtype = recording.get_dtype()
        # the "apply_shift" function returns a float64 buffer. In case the dtype is different
        # than float64, we need a temporary casting and to force the buffer back to the original dtype
        if str(dtype) != "float64":
            tmp_dtype = np.dtype("float64")
        else:
            tmp_dtype = None

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        for parent_segment in recording._recording_segments:
            rec_segment = PhaseShiftRecordingSegment(parent_segment, sample_shifts, margin, dtype, tmp_dtype)
            self.add_recording_segment(rec_segment)

        # for dumpability
        if inter_sample_shift is not None:
            inter_sample_shift = list(inter_sample_shift)
        self._kwargs = dict(recording=recording, margin_ms=float(margin_ms), inter_sample_shift=inter_sample_shift)


class PhaseShiftRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, sample_shifts, margin, dtype, tmp_dtype):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.sample_shifts = sample_shifts
        self.margin = margin
        self.dtype = dtype
        self.tmp_dtype = tmp_dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)

        # this return a copy with margin  + taper on border always
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            channel_indices,
            self.margin,
            dtype=self.tmp_dtype,
            add_zeros=True,
            window_on_margin=True,
        )
        traces_shift = apply_frequency_shift(traces_chunk, self.sample_shifts[channel_indices], axis=0)

        traces_shift = traces_shift[left_margin:-right_margin, :]
        if self.tmp_dtype is not None:
            if np.issubdtype(self.dtype, np.integer):
                traces_shift = traces_shift.round()
            traces_shift = traces_shift.astype(self.dtype)

        return traces_shift


# function for API
phase_shift = define_function_handling_dict_from_class(source_class=PhaseShiftRecording, name="phase_shift")


def apply_frequency_shift(signal, shift_samples, axis=0):
    """
    Apply frequency shift to a signal buffer. This allow for shifting that are sub-sample accurate.

    Parameters
    ----------
    signal : ndarray
        Input signal array to be shifted.
    shift_samples : ndarray
        Array of sample shifts for each channel. Phase shifts are in units of 1/sampling_rate.
    axis : int, optional
        Axis along which to perform the shift. Currently, only axis=0 is supported.

    Returns
    -------
    shifted_signal : ndarray
        Signal array with the applied frequency shifts.

    Notes
    -----
    The function works by transforming the signal to the frequency domain using the real FFT (rFFT),
    applying phase shifts, and then transforming back to the time domain using the inverse real FFT (irFFT).
    The phase shifts are calculated based on the frequency grid obtained from the FFT.

    The key steps are:
    1. Compute the rFFT of the input signal.
    2. Calculate the frequency grid and use it to compute the phase shifts.
    3. Apply the phase shifts in the frequency domain.
    4. Perform the inverse rFFT to obtain the shifted signal in the time domain.

    This method leverages the properties of the Fourier transform, where a phase shift in the frequency domain
    corresponds to a time shift in the time domain.
    """
    import scipy.fft

    signal_length = signal.shape[axis]
    num_channels = shift_samples.size
    fourier_signal_size = signal_length // 2 + 1

    frequency_domain_signal = scipy.fft.rfft(signal, n=signal_length, axis=axis, overwrite_x=True)
    fourier_signal_size = frequency_domain_signal.shape[0]

    if axis == 0:
        frequency_grid = np.empty(shape=(fourier_signal_size, num_channels))
        # Note that np.fft.rfttfreq handles both even and odd signal lengths
        frequency_grid[:, :] = 2 * np.pi * np.fft.rfftfreq(signal_length)[:, np.newaxis]
        shifts = np.multiply(frequency_grid, shift_samples[np.newaxis, :], out=frequency_grid)
    else:
        raise NotImplementedError("Axis != 0 is not implemented yet")

    # Rotate the signal in the frequency domain
    rotations = np.exp(-1j * shifts)
    phase_shifted_signal = np.multiply(frequency_domain_signal, rotations, out=rotations)

    # Inverse FFT to get the translated signal
    shifted_signal = scipy.fft.irfft(phase_shifted_signal, n=signal_length, axis=axis, overwrite_x=True)
    return shifted_signal


apply_fshift = apply_frequency_shift


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
    from scipy.fft import rfft, irfft

    # create a vector that contains a 1 sample shift on the axis
    ns = ns or w.shape[axis]
    shape = np.array(w.shape) * 0 + 1
    shape[axis] = ns
    dephas = np.zeros(shape)
    # np.put(dephas, 1, 1)
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
