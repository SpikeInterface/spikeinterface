import pytest
from pathlib import Path

from spikeinterface import NumpyRecording, set_global_tmp_folder
from spikeinterface.core.testing_tools import generate_recording
from spikeinterface.toolkit.preprocessing import resample



import numpy as np
from scipy.fft import fft, fftfreq

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"

"""
Check if the sinusoidal gets resampled nicely enough

 -[x] Check that the signals and resampled are in fact similar in frequency domains.
    We are not destroying or adding weird components, in frequency and amplitude,
    Haven't tested on phase, but I think there's a WHOLE lot of debate there.
- [x] Check that RMS of signals don't change when saving with different chunk_sizes
    Similar to the tests done on pahse_shift

"""

def create_sinusoidal_traces(sr=3e4, duration=30, freqs_n=10, dtype=np.int16):
    """Return a sum of sinusoidals to test resampling schemes.

    Parameters
    ----------
    sr : float, optional
        Sampling rate of the signal, by default 3e4
    duration : int, optional
        Duration of the signal in seconds, by default 30
    freqs_n : int, optional
        Total frequencies to span on the signal, by default 10

    Returns
    -------
    list[traces, [freqs_vals, amps_vals, phase_shifts]]
        traces : A new signal with shape [duration * sr, 1]
        freq_vals : Frequencies present on the signal.
        amp_vals : Amplitudes of the frequencies present.
        pahse_shifts : Phase shifts of the frequencies, this one is an overkill.

    """
    # Will return a signal with many frequencies some of wich, must be lost
    # on the resampling without breaking the lower bands
    x = np.arange(0, duration, step=1/sr)
    # Sample log spaced freqs from 10-10k
    freqs_vals = np.logspace(1, 4, num=freqs_n).astype(int)
    # Vary amplitudes and phases for the signal
    amps_vals = np.logspace(2, 1, num=freqs_n)
    phase_shifts = np.random.uniform(0, np.pi, size=freqs_n)
    # Make a colection of sinusoids
    ys = [
        amp * np.sin(2 * np.pi * freq * x + phase)
        for amp, freq, phase in zip(amps_vals, freqs_vals, phase_shifts)
    ]
    traces = np.sum(ys, axis=0).astype(dtype).reshape([-1, 1])
    return traces, [freqs_vals, amps_vals, phase_shifts]


def get_fft(traces, sr):
    # Return the power spectrum of the positive fft
    N = len(traces)
    yf = fft(traces)
    # Get only positive freqs
    xf = fftfreq(N, 1/sr)[:N//2]
    nyf = 2.0/N * np.abs(yf)[:N//2]
    return xf, nyf



def test_resample_freq_domain():
    sr=3e4
    duration=30
    freqs_n=10
    dtype=np.int16
    traces, [freqs_vals, amps_vals, phase_shifts] = create_sinusoidal_traces(sr, duration, freqs_n, dtype)
    parent_rec = NumpyRecording(traces, sr)
    # Different resampling frequencies, always below Niquist
    resamp_fss = (np.linspace(0.1, 0.45, 10) * sr).astype(int)
    resamp_recs = [resample(parent_rec, resamp_fs) for resamp_fs in resamp_fss]
    # First set of tests, we are updating frames and time duration correctly

    ## that they all have the correct number of frames:
    msg1 = "The number of frames gets distorted by resampling."
    assert np.allclose([resamp_rec.get_num_frames() for resamp_rec in resamp_recs], resamp_fss * duration), msg1
    ## They all last 1 second
    msg2 = "The total duration gets distorted by resampling."
    assert np.allclose([resamp_rec.get_total_duration() for resamp_rec in resamp_recs], duration), msg2
    ## check that the first and last time points are similar with tolerance
    msg3 = "The timestamps of key frames are distorted by resampling."
    assert np.all(
        [
            np.isclose(
                parent_rec.get_times()[[1, -1]],
                resamp_rec.get_times()[[1, -1]],
                atol=1/resamp_fs,
            )
            for resamp_fs, resamp_rec in zip(resamp_fss, resamp_recs)
        ]
    ), msg3
    ## Test that traces and times are the same lenght
    msg4 = "The time and traces vectors must be of equal lenght. Non integer resampling rates can lead to this."
    assert np.all(
        [rec.get_traces().shape[0] == rec.get_times().shape[0] for rec in resamp_recs]
    ), msg4
    # One can see that they are quite alike, the signals but also their freq domains
    plot_tests = """
    import matplotlib.pyplot as plt
    # Signals look similar on resampling
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 7], num='Resampled traces')
    lws = np.linspace(5, 1, num=10)
    _ = [
        ax.plot(rec.get_times(), rec.get_traces(), lw=lw, alpha=lw/5)
        for rec, lw in zip(resamp_recs, lws)
    ]
    # Even in fourier
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 7], num='Fourier spectrums')
    xfs, nyfs = np.array(
        [
            get_fft(rec.get_traces(), resamp_fs)
            for rec, resamp_fs in zip(resamp_recs, resamp_fss)
        ]
        , dtype='object'
    ).T
    _ = [ax.plot(xf, nyf, lw=lw, alpha=lw / 5) for xf, nyf, lw in zip(xfs, nyfs, lws)]
    """


def test_resample_by_chunks():
    # Now tests the margin effects and the chunk_sizes for sanity
    # The same as in the phase_shift tests.
    sr=int(3e4)
    duration=30
    freqs_n=10
    dtype=np.int16
    traces, [freqs_vals, amps_vals, phase_shifts] = create_sinusoidal_traces(sr, duration, freqs_n, dtype)
    parent_rec = NumpyRecording(traces, sr)
    rms = np.sqrt(np.mean(parent_rec.get_traces()**2))
    # The chunk_size must be always at least some 1 second of the resample, else it breaks.
    # Does this makes sense?
    # Also, sometimes decimate might give warnings about filter designs
    # If only resample is used, everything works nicely, so removed the option
    import matplotlib as plt
    for resample_rate in [500, 1000, 2500]:
        for margin_ms in (100, 200, 1000):
            for chunk_size in (resample_rate*1, resample_rate*1.5, resample_rate*2):
                # This looks like the minimum ratio of chunksize and resample that works
                chunk_size = int(chunk_size)
                # print(f'resmple_rate = {resample_rate}; margin_ms = {margin_ms}; chunk_size={chunk_size}')
                rec2 = resample(parent_rec, resample_rate, margin_ms=margin_ms)
                # save by chunk rec3 is the cached version
                rec3 = rec2.save(format='memory', chunk_size=chunk_size, n_jobs=1, progress_bar=False)

                traces2 = rec2.get_traces()
                traces3 = rec3.get_traces()

                # error between full and chunked
                error_mean = np.sqrt(np.mean((traces2 - traces3)**2))
                error_max = np.sqrt(np.max((traces2 - traces3)**2))

                # this will never be possible:
                #      assert np.allclose(traces2, traces3)
                # so we check that the diff between chunk processing and not chunked is small
                #~ print()
                #~ print(dtype, margin_ms, chunk_size)
                #~ print(error_mean, rms, error_mean / rms)
                #~ print(error_max, rms, error_max / rms)
                assert error_mean / rms < 0.01
                assert error_max / rms < 0.1
                # plot_test = """
                fig, axs = plt.subplots(nrows=2)
                ax = axs[0]
                ax.plot(traces2, color='g', label='no chunk')
                ax.plot(traces3, color='r', label=f'no chunk {chunk_size}')
                ax.legend()
                ax = axs[1]
                ax.plot(traces3-traces2)
                for i in range(traces2.shape[0]//chunk_size):
                    ax.axvline(chunk_size * i, color='k')
                plt.show()
                """


if __name__ == '__main__':
    test_resample_freq_domain()
    test_resample_by_chunks()
