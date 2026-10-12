import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class

from spikeinterface.core import get_chunk_with_margin, apply_raised_cosine_taper

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

# 16 taps keep the response flat across the Neuropixels passband; fewer start to cut into it.
_DEFAULT_FIR_TAPS = 16


class PhaseShiftRecording(BasePreprocessor):
    """
    Apply phase shifts to compensate for the staggered ADC sampling times in Neuropixels recordings.

    This code is inspired from from  IBL lib.
    https://github.com/int-brain-lab/ibllib/blob/master/ibllib/dsp/fourier.py
    and also the one from spikeglx
    https://billkarsh.github.io/SpikeGLX/help/dmx_vs_gbl/dmx_vs_gbl/

    Parameters
    ----------
    recording : Recording
        The recording. It need to have  "inter_sample_shift" in properties.
    margin_ms : float, default: 40.0
        Margin in ms read on each side of a chunk with ``method="fft"``; 40 ms keeps the error at chunk edges small.
        Ignored with ``method="fir"``.
    inter_sample_shift : None or numpy array, default: None
        If "inter_sample_shift" is not in recording properties,
        we can externally provide one.
    dtype : None | str | dtype, default: None
        Dtype of the output. None keeps the parent's.
    method : "fft" | "fir", default: "fft"
        - ``"fft"``: shifts each chunk in the frequency domain. Exact up to Nyquist.
        - ``"fir"``: convolves each channel with a short windowed-sinc filter. Faster, and with the default
          ``n_taps`` flat up to about a third of the sampling rate, which covers the Neuropixels passband. For
          signal above that, raise ``n_taps`` or use ``"fft"``.
    n_taps : int, default: 16
        Filter length with ``method="fir"``; must be even. Longer filters stay flat to higher frequencies and run
        slower. Ignored with ``method="fft"``.

    Returns
    -------
    filter_recording : PhaseShiftRecording
        The phase shifted recording object
    """

    def __init__(
        self,
        recording,
        margin_ms=40.0,
        inter_sample_shift=None,
        dtype=None,
        method="fft",
        n_taps=_DEFAULT_FIR_TAPS,
    ):
        if method not in ("fft", "fir"):
            raise ValueError(f"method must be 'fft' or 'fir', got {method!r}")
        if method == "fir":
            if n_taps < 2 or n_taps % 2 != 0:
                raise ValueError(f"n_taps must be a positive even integer, got {n_taps}")

        if inter_sample_shift is None:
            assert "inter_sample_shift" in recording.get_property_keys(), "'inter_sample_shift' is not a property!"
            sample_shifts = recording.get_property("inter_sample_shift")
        else:
            assert (
                len(inter_sample_shift) == recording.get_num_channels()
            ), "the 'inter_sample_shift' must be same size at the num_channels "
            sample_shifts = np.asarray(inter_sample_shift)

        if dtype is None:
            dtype = recording.get_dtype()

        if method == "fft":
            margin = int(margin_ms * recording.get_sampling_frequency() / 1000.0)
            # apply_frequency_shift computes in float64; other dtypes are cast back after it.
            tmp_dtype = np.dtype("float64") if str(dtype) != "float64" else None
        else:
            margin = n_taps // 2
            tmp_dtype = None

        advertised_dtype = np.dtype(dtype)

        BasePreprocessor.__init__(self, recording, dtype=advertised_dtype)
        for parent_segment in recording.segments:
            rec_segment = PhaseShiftRecordingSegment(
                parent_segment,
                sample_shifts,
                margin,
                advertised_dtype,
                tmp_dtype,
                method=method,
                n_taps=n_taps,
            )
            self.add_recording_segment(rec_segment)

        # for dumpability
        if inter_sample_shift is not None:
            inter_sample_shift = list(inter_sample_shift)
        self._kwargs = dict(
            recording=recording,
            margin_ms=float(margin_ms),
            inter_sample_shift=inter_sample_shift,
            dtype=advertised_dtype.str,
            method=method,
            n_taps=int(n_taps),
        )


class PhaseShiftRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        sample_shifts,
        margin,
        dtype,
        tmp_dtype,
        method="fft",
        n_taps=_DEFAULT_FIR_TAPS,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.sample_shifts = np.asarray(sample_shifts)
        self.margin = margin
        self.dtype = np.dtype(dtype)
        self.tmp_dtype = tmp_dtype
        self.method = method
        self.n_taps = int(n_taps)
        self._fir_kernels_kc = None
        if method == "fir":
            # The kernels correlate, out[n] += kernels_kc[k] * x[n - half + k], in (K, C) so the channel loop is
            # contiguous.
            taps = _build_fir_taps(self.sample_shifts, self.n_taps)
            self._fir_kernels_kc = np.ascontiguousarray(taps[:, ::-1].T, dtype=np.float32)

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)
        if self.method == "fir":
            return self._get_traces_fir(start_frame, end_frame, channel_indices)
        return self._get_traces_fft(start_frame, end_frame, channel_indices)

    def _get_traces_fft(self, start_frame, end_frame, channel_indices):
        # A float64 parent can return a view of its own data, so pin the dtype to get a copy the taper can modify.
        compute_dtype = self.tmp_dtype if self.tmp_dtype is not None else np.dtype("float64")
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            channel_indices,
            self.margin,
            dtype=compute_dtype,
            add_zeros=True,
            window_on_margin=False,
        )
        apply_raised_cosine_taper(traces_chunk, self.margin, inplace=True)
        traces_shift = apply_frequency_shift(traces_chunk, self.sample_shifts[channel_indices], axis=0)

        traces_shift = traces_shift[left_margin:-right_margin, :]
        if np.issubdtype(self.dtype, np.integer):
            traces_shift = traces_shift.round()
        if traces_shift.dtype != self.dtype:
            traces_shift = traces_shift.astype(self.dtype)

        return traces_shift

    def _get_traces_fir(self, start_frame, end_frame, channel_indices):
        half = self.n_taps // 2
        parent = self.parent_recording_segment
        length = int(parent.get_num_samples())
        fetch_start = max(0, start_frame - half)
        fetch_end = min(length, end_frame + half)
        left_pad = half - (start_frame - fetch_start)
        right_pad = half - (fetch_end - end_frame)

        traces = parent.get_traces(start_frame=fetch_start, end_frame=fetch_end, channel_indices=channel_indices)

        if left_pad > 0 or right_pad > 0:
            full_len = (end_frame - start_frame) + 2 * half
            padded = np.zeros((full_len, traces.shape[1]), dtype=traces.dtype)
            padded[left_pad : left_pad + traces.shape[0], :] = traces
            traces = padded

        if isinstance(channel_indices, slice) and channel_indices == slice(None):
            kernels_kc = self._fir_kernels_kc
        else:
            kernels_kc = np.ascontiguousarray(self._fir_kernels_kc[:, channel_indices])

        if not _HAS_NUMBA:
            shifted = _sinc_fir_numpy_tc(traces, kernels_kc)
        elif traces.dtype == np.int16:
            traces = np.ascontiguousarray(traces)
            shifted = _sinc_fir_kernel_int16_tc(traces, kernels_kc)
        else:
            sig_f32 = np.ascontiguousarray(traces, dtype=np.float32)
            shifted = _sinc_fir_kernel_tc(sig_f32, kernels_kc)

        out = shifted[half : half + (end_frame - start_frame), :]

        if out.dtype == self.dtype:
            return out
        if np.issubdtype(self.dtype, np.integer):
            return out.round().astype(self.dtype)
        return out.astype(self.dtype)


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
    from scipy.fft import irfft, next_fast_len, rfft

    signal_length = signal.shape[axis]
    # A length with a large prime factor makes the FFT slow. Zero padding after the tapered end
    # changes only the period of the circular shift, which moves the output by far less than one int16 step.
    fft_length = next_fast_len(signal_length, real=True)
    frequency_domain_signal = rfft(signal, n=fft_length, axis=axis, overwrite_x=True)

    if axis == 0:
        angular_frequencies = 2 * np.pi * np.fft.rfftfreq(fft_length)
        # Neuropixels channels reuse a small set of ADC sampling delays, so compute each rotation once.
        unique_shifts, shift_indices = np.unique(shift_samples, return_inverse=True)
        unique_angles = (angular_frequencies[:, np.newaxis] * unique_shifts[np.newaxis, :]).astype(
            np.float64, copy=False
        )
        unique_rotations = np.exp(-1j * unique_angles)
        rotations = unique_rotations[:, shift_indices]
        phase_shifted_signal = np.multiply(frequency_domain_signal, rotations, out=rotations)
    else:
        raise NotImplementedError("Axis != 0 is not implemented yet")

    # Inverse FFT to get the translated signal
    shifted_signal = irfft(phase_shifted_signal, n=fft_length, axis=axis, overwrite_x=True)
    return shifted_signal[:signal_length]


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


def _build_fir_taps(shift_samples, n_taps, beta=8.6):
    """Per-channel Kaiser-windowed sinc kernels that delay each channel by its fractional shift, as ``(C, K)``.

    Rows are in ``fir_filter``'s tap order, ``y[n] = sum_j taps[j] * x[n + center - j]`` with its default
    ``center = (K - 1) // 2``. A positive shift delays, ``y[n] = x[n - shift]``, as in ``apply_frequency_shift``.
    ``beta=8.6`` gives about -80 dB sidelobes.
    """
    # Tap j weighs x[n + center - j], an offset of t = center - j samples, so the ideal delay gives sinc(t + shift).
    offsets = (n_taps - 1) // 2 - np.arange(n_taps, dtype=np.float64)
    shifts = np.asarray(shift_samples, dtype=np.float64)[:, np.newaxis]
    taps = np.sinc(offsets + shifts) * np.kaiser(n_taps, beta=beta)
    # Unit DC gain: the windowed sinc's sum varies with the shift, which would scale each ADC group differently.
    taps /= taps.sum(axis=1, keepdims=True)
    return taps


try:
    import numba

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - the FIR path falls back to _sinc_fir_numpy_tc
    _HAS_NUMBA = False


if _HAS_NUMBA:
    # Single-threaded: chunk workers supply the parallelism, and parallel=True crashes under numba's workqueue
    # layer with the thread engine and under its omp layer with forked processes.
    @numba.njit(nogil=True, cache=True, boundscheck=False)
    def _sinc_fir_kernel_tc(signal_tc, kernels_kc):
        """Per-channel FIR on a ``(T, C)`` float32 signal, treating samples outside it as zero."""
        T, C = signal_tc.shape
        K = kernels_kc.shape[0]
        half = K // 2
        interior_start = half
        interior_end = T - (K - 1 - half)
        # np.zeros would defer zeroing to page faults inside the loop; zeroing each row just before use keeps it in cache
        out = np.empty((T, C), dtype=np.float32)
        for n in range(T):
            out[n] = 0.0
            if interior_start <= n < interior_end:
                base = n - half
                for k in range(K):
                    for c in range(C):
                        out[n, c] += kernels_kc[k, c] * signal_tc[base + k, c]
            else:
                for k in range(K):
                    idx = n + k - half
                    if 0 <= idx < T:
                        for c in range(C):
                            out[n, c] += kernels_kc[k, c] * signal_tc[idx, c]
        return out

    @numba.njit(nogil=True, cache=True, boundscheck=False)
    def _sinc_fir_kernel_int16_tc(signal_tc, kernels_kc):
        """int16-input variant: reads int16 and writes float32, so an int16 parent needs no cast first."""
        T, C = signal_tc.shape
        K = kernels_kc.shape[0]
        half = K // 2
        interior_start = half
        interior_end = T - (K - 1 - half)
        # np.zeros would defer zeroing to page faults inside the loop; zeroing each row just before use keeps it in cache
        out = np.empty((T, C), dtype=np.float32)
        for n in range(T):
            out[n] = 0.0
            if interior_start <= n < interior_end:
                base = n - half
                for k in range(K):
                    for c in range(C):
                        out[n, c] += kernels_kc[k, c] * np.float32(signal_tc[base + k, c])
            else:
                for k in range(K):
                    idx = n + k - half
                    if 0 <= idx < T:
                        for c in range(C):
                            out[n, c] += kernels_kc[k, c] * np.float32(signal_tc[idx, c])
        return out


def _sinc_fir_numpy_tc(signal_tc, kernels_kc):
    """Numpy equivalent of the numba kernels, used when numba is not installed.

    Accumulates one tap at a time over the whole ``(T, C)`` block, in the same tap order as the numba kernels,
    so the two agree to float32 rounding.
    """
    signal = np.asarray(signal_tc, dtype=np.float32)
    T = signal.shape[0]
    K = kernels_kc.shape[0]
    half = K // 2
    out = np.zeros(signal.shape, dtype=np.float32)
    product = np.empty(signal.shape, dtype=np.float32)
    for k in range(K):
        offset = k - half  # out[n] += h[k] * x[n + offset], zero outside [0, T)
        n0 = max(0, -offset)
        n1 = min(T, T - offset)
        if n1 <= n0:
            continue
        np.multiply(signal[n0 + offset : n1 + offset], kernels_kc[k], out=product[n0:n1])
        out[n0:n1] += product[n0:n1]
    return out
