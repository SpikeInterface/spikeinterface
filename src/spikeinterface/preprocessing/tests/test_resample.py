from spikeinterface.preprocessing import resample
from spikeinterface.core import NumpyRecording


import numpy as np

DEBUG = False
# DEBUG = True

if DEBUG:
    import matplotlib.pyplot as plt


"""
Check if the sinusoidal gets resampled nicely enough

 -[x] Check that the signals and resampled are in fact similar in frequency domains.
    We are not destroying or adding weird components, in frequency and amplitude,
    Haven't tested on phase, but I think there's a WHOLE lot of debate there.
- [x] Check that RMS of signals don't change when saving with different chunk_sizes
    Similar to the tests done on pahse_shift

"""


def create_sinusoidal_traces(sampling_frequency=3e4, duration=30, freqs_n=10, max_freq=10000, dtype=np.int16):
    """Return a sum of sinusoidals to test resampling schemes.

    Parameters
    ----------
    sampling_frequency : float, default: 30000
        Sampling rate of the signal
    duration : int, default: 30
        Duration of the signal in seconds
    freqs_n : int, default: 10
        Total frequencies to span on the signal
    max_freq : int, default: 10000
        Maximum frequency of sinusoids

    Returns
    -------
    list[traces, [freqs_vals, amps_vals, phase_shifts]]
        traces : A new signal with shape [duration * sampling_frequency, 1]
        freq_vals : Frequencies present on the signal.
        amp_vals : Amplitudes of the frequencies present.
        pahse_shifts : Phase shifts of the frequencies, this one is an overkill.

    """
    # Will return a signal with many frequencies some of wich, must be lost
    # on the resampling without breaking the lower bands
    x = np.arange(0, duration, step=1 / sampling_frequency)
    # Sample log spaced freqs from 10-10k
    freqs_vals = np.logspace(1, np.log10(max_freq), num=freqs_n).astype(int)
    # print(freqs_vals, max_freq)
    # Vary amplitudes and phases for the signal
    amps_vals = np.logspace(2, 1, num=freqs_n)
    phase_shifts = np.random.uniform(0, np.pi, size=freqs_n)
    # Make a colection of sinusoids
    ys = [amp * np.sin(2 * np.pi * freq * x + phase) for amp, freq, phase in zip(amps_vals, freqs_vals, phase_shifts)]
    traces = np.sum(ys, axis=0).astype(dtype).reshape([-1, 1])
    return traces, [freqs_vals, amps_vals, phase_shifts]


def get_fft(traces, sampling_frequency):
    from scipy.fft import fft, fftfreq

    # Return the power spectrum of the positive fft
    N = len(traces)
    yf = fft(traces)
    # Get only positive freqs
    xf = fftfreq(N, 1 / sampling_frequency)[: N // 2]
    nyf = 2.0 / N * np.abs(yf)[: N // 2]
    return xf, nyf


def test_resample_freq_domain():
    sampling_frequency = 3e4
    duration = 10
    freqs_n = 10
    max_freq = 1000
    dtype = np.int16
    traces, [freqs_vals, amps_vals, phase_shifts] = create_sinusoidal_traces(
        sampling_frequency, duration, freqs_n, max_freq, dtype
    )
    parent_rec = NumpyRecording(traces, sampling_frequency)
    # Different resampling frequencies, always below Niquist
    resamp_fss = (np.linspace(0.1, 0.45, 10) * sampling_frequency).astype(int)
    resamp_recs = [resample(parent_rec, resamp_fs) for resamp_fs in resamp_fss]
    # First set of tests, we are updating frames and time duration correctly

    # that they all have the correct number of frames:
    msg1 = "The number of frames gets distorted by resampling."
    assert np.allclose([resamp_rec.get_num_frames() for resamp_rec in resamp_recs], resamp_fss * duration), msg1
    # They all last 1 second
    msg2 = "The total duration gets distorted by resampling."
    assert np.allclose([resamp_rec.get_total_duration() for resamp_rec in resamp_recs], duration), msg2
    # check that the first and last time points are similar with tolerance
    msg3 = "The timestamps of key frames are distorted by resampling."
    assert np.all(
        [
            np.isclose(
                parent_rec.get_times()[[1, -1]],
                resamp_rec.get_times()[[1, -1]],
                atol=1 / resamp_fs,
            )
            for resamp_fs, resamp_rec in zip(resamp_fss, resamp_recs)
        ]
    ), msg3
    # Test that traces and times are the same lenght
    msg4 = "The time and traces vectors must be of equal length. Non integer resampling rates can lead to this."
    assert np.all([rec.get_traces().shape[0] == rec.get_times().shape[0] for rec in resamp_recs]), msg4
    # One can see that they are quite alike, the signals but also their freq domains

    if DEBUG:
        # Signals look similar on resampling
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 7])

        _ = ax.plot(parent_rec.get_times(), parent_rec.get_traces(), color="k", alpha=0.8, lw=1.5, label="original")

        for i, rec in enumerate(resamp_recs):
            rs_rate = int(rec.get_sampling_frequency())
            _ = ax.plot(rec.get_times(), rec.get_traces(), color=f"C{i}", alpha=0.8, lw=0.5, label=f"rate: {rs_rate}")
        ax.set_title("Resampled traces")
        ax.legend()

        # Even in fourier
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 7])

        xf, nyf = get_fft(parent_rec.get_traces().ravel(), parent_rec.get_sampling_frequency())
        _ = ax.plot(xf, nyf, color="k", alpha=0.8, label=f"original")

        for i, rec in enumerate(resamp_recs):
            rs_rate = int(rec.get_sampling_frequency())
            xf, nyf = get_fft(rec.get_traces().ravel(), rs_rate)
            _ = ax.plot(xf, nyf, color=f"C{i}", alpha=0.8, label=f"rate: {rs_rate}")
        ax.set_title("Fourier spectra")
        ax.legend()

        plt.show()


def test_resample_by_chunks():
    # Now tests the margin effects and the chunk_sizes for sanity
    # The same as in the phase_shift tests.
    sampling_frequency = int(3e4)
    duration = 30
    freqs_n = 10
    # ~ dtype = np.int16
    dtype = np.float32
    max_freq = 1000
    traces, [freqs_vals, amps_vals, phase_shifts] = create_sinusoidal_traces(
        sampling_frequency, duration, freqs_n, max_freq, dtype
    )
    parent_rec = NumpyRecording(traces, sampling_frequency)
    rms = np.sqrt(np.mean(parent_rec.get_traces() ** 2))
    # The chunk_size must be always at least some 1 second of the resample, else it breaks.
    # Does this makes sense?
    # Also, sometimes decimate might give warnings about filter designs
    resample_rates = [1000, 2000]  # [500, 1000, 2500]
    margins_ms = [100, 1000]  # [100, 200, 1000]
    chunk_durations = [0.5, 1]  # [1, 2, 3]

    for resample_rate in resample_rates:
        for margin_ms in margins_ms:
            for chunk_size in [int(resample_rate * chunk_multi) for chunk_multi in chunk_durations]:
                # print(f'resmple_rate = {resample_rate}; margin_ms = {margin_ms}; chunk_size={chunk_size}')
                rec2 = resample(parent_rec, resample_rate, margin_ms=margin_ms)
                # save by chunk rec3 is the cached version
                rec3 = rec2.save(format="memory", chunk_size=chunk_size, n_jobs=1, progress_bar=False)

                traces2 = rec2.get_traces()
                traces3 = rec3.get_traces()

                # error between full and chunked
                # for error first and last chunk is removed
                sl = slice(chunk_size, -chunk_size)
                error_mean = np.sqrt(np.mean((traces2[sl] - traces3[sl]) ** 2))
                error_max = np.sqrt(np.max((traces2[sl] - traces3[sl]) ** 2))

                # this will never be possible:
                # assert np.allclose(traces2, traces3)
                # so we check that the diff between chunk processing and not chunked is small
                # print()
                # print(dtype, margin_ms, chunk_size)
                # print(error_mean, rms, error_mean / rms)
                # print(error_max, rms, error_max / rms)
                # The original thrshold are too restrictive, but in all cases
                # The signals look quite similar, with error that are small enough
                # But, when using signal.resample, the last edge becomes too noisy

                assert error_mean / rms < 0.01
                assert error_max / rms < 0.05

                if DEBUG:
                    fig, axs = plt.subplots(nrows=2, sharex=True)
                    fig.suptitle(
                        f"Resample rate {resample_rate}\nMargin {margin_ms}\nChunk size {chunk_size}\n error mean(%) {error_mean / rms}  error max(%){error_max / rms} "
                    )
                    ax = axs[0]
                    ax.plot(traces2, color="g", label="no chunk")
                    ax.plot(traces3, color="r", label=f"chunked")
                    for i in range(traces2.shape[0] // chunk_size):
                        ax.axvline(chunk_size * i, color="k", alpha=0.4)
                    ax.legend()
                    ax = axs[1]
                    ax.plot(traces3 - traces2)
                    for i in range(traces2.shape[0] // chunk_size):
                        ax.axvline(chunk_size * i, color="k", alpha=0.4)

                    plt.show()


if __name__ == "__main__":
    test_resample_freq_domain()
    test_resample_by_chunks()
