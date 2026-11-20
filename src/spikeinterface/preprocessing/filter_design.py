"""Filter design, separate from filter application, so kernels can be inspected and composed before use."""

from __future__ import annotations

import numpy as np


def _stopband_db(taps, fs, edges):
    """Smallest attenuation, in dB, of `taps` over the frequency intervals `edges`, sampled 16 times per ripple."""
    import scipy.signal

    worst = 0.0
    for low, high in edges:
        freqs = np.linspace(low, high, max(64, int(16 * (high - low) / fs * taps.size)))
        _, response = scipy.signal.freqz(taps, worN=freqs, fs=fs)
        worst = max(worst, np.abs(response).max())
    return -20 * np.log10(max(worst, 1e-300))


def design_fir_bandpass(freq_min, freq_max, fs, transition_low_hz=175.0, transition_high_hz=2000.0, stopband_db=45.0):
    """
    Design a linear-phase bandpass FIR kernel by the Kaiser window method.

    The response is -6 dB at `freq_min` and `freq_max`, and at least `stopband_db` down below
    ``freq_min - transition_low_hz / 2`` and above ``freq_max + transition_high_hz / 2``. Passband ripple is at
    most about ``2 * 10 ** (-stopband_db / 20)`` in amplitude. The kernel convolves a highpass and a lowpass,
    each windowed to its own transition: one window would give both edges the narrower width, and a sharp
    upper edge rings for longer.

    Parameters
    ----------
    freq_min, freq_max : float
        Lower and upper -6 dB cutoffs, in Hz.
    fs : float
        Sampling frequency, in Hz.
    transition_low_hz, transition_high_hz : float, default: 175.0, 2000.0
        Width of the transition band centred on each cutoff, in Hz.
    stopband_db : float, default: 45.0
        Least attenuation in both stopbands, in dB.

    Returns
    -------
    taps : np.ndarray
        Symmetric, odd-length float64 kernel, for ``fir_filter``.
    """
    import scipy.signal

    stop_low = freq_min - transition_low_hz / 2
    stop_high = freq_max + transition_high_hz / 2
    if transition_low_hz <= 0 or transition_high_hz <= 0 or stopband_db <= 0:
        raise ValueError("transition_low_hz, transition_high_hz and stopband_db must be positive")
    if not 0 < stop_low < freq_min < freq_max < stop_high < fs / 2:
        raise ValueError(
            f"The transition bands, [{stop_low}, {freq_min + transition_low_hz / 2}] and "
            f"[{freq_max - transition_high_hz / 2}, {stop_high}] Hz, must lie in (0, {fs / 2}) Hz in that order"
        )

    def edge(cutoff, width, pass_zero):
        num_taps, beta = scipy.signal.kaiserord(stopband_db, width / (fs / 2))
        return num_taps | 1, beta, cutoff, pass_zero

    edges = [edge(freq_min, transition_low_hz, False), edge(freq_max, transition_high_hz, True)]
    stopbands = [(0.0, stop_low), (stop_high, fs / 2)]
    while True:
        highpass, lowpass = (
            scipy.signal.firwin(n, cutoff, window=("kaiser", beta), pass_zero=pass_zero, fs=fs)
            for n, beta, cutoff, pass_zero in edges
        )
        taps = np.convolve(highpass, lowpass)
        # The two kernels' ripples can add, so the window's nominal attenuation is checked and the failing edge
        # lengthened.
        misses = [_stopband_db(taps, fs, [band]) < stopband_db for band in stopbands]
        if not any(misses):
            return taps
        edges = [(n + 2 * miss, beta, cutoff, pz) for (n, beta, cutoff, pz), miss in zip(edges, misses)]


def design_fir_highpass(freq_min, fs, transition_low_hz=175.0, stopband_db=45.0):
    """
    Design a linear-phase highpass FIR kernel by the Kaiser window method.

    The response is -6 dB at `freq_min` and at least `stopband_db` down below ``freq_min - transition_low_hz / 2``,
    with passband ripple of at most about ``10 ** (-stopband_db / 20)`` in amplitude.

    Parameters
    ----------
    freq_min : float
        The -6 dB cutoff, in Hz.
    fs : float
        Sampling frequency, in Hz.
    transition_low_hz : float, default: 175.0
        Width of the transition band centred on the cutoff, in Hz.
    stopband_db : float, default: 45.0
        Least attenuation in the stopband, in dB.

    Returns
    -------
    taps : np.ndarray
        Symmetric, odd-length float64 kernel, for ``fir_filter``.
    """
    import scipy.signal

    stop = freq_min - transition_low_hz / 2
    if transition_low_hz <= 0 or stopband_db <= 0:
        raise ValueError("transition_low_hz and stopband_db must be positive")
    if not 0 < stop < freq_min + transition_low_hz / 2 < fs / 2:
        raise ValueError(
            f"The transition band, [{stop}, {freq_min + transition_low_hz / 2}] Hz, must lie in (0, {fs / 2}) Hz"
        )
    num_taps, beta = scipy.signal.kaiserord(stopband_db, transition_low_hz / (fs / 2))
    num_taps |= 1  # a highpass needs a type I kernel
    while True:
        taps = scipy.signal.firwin(num_taps, freq_min, window=("kaiser", beta), pass_zero=False, fs=fs)
        if _stopband_db(taps, fs, [(0.0, stop)]) >= stopband_db:
            return taps
        num_taps += 2
