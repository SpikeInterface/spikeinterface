from spikeinterface.preprocessing import resample
from spikeinterface.core import NumpyRecording


import numpy as np
import pytest

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


def _make_gapped_recording(sampling_frequency=30000, n_channels=2, sec1_duration=1.0, sec2_duration=1.0, gap_s=5.0):
    """Helper: create a NumpyRecording with a time_vector gap between two sections."""
    n1 = int(sec1_duration * sampling_frequency)
    n2 = int(sec2_duration * sampling_frequency)
    n_total = n1 + n2
    traces = np.random.randn(n_total, n_channels).astype(np.float32)
    rec = NumpyRecording(traces, sampling_frequency)

    tv = np.arange(n_total, dtype="float64") / sampling_frequency
    tv[n1:] += gap_s
    rec.set_times(tv)
    return rec, n1, n2


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


def test_resample_preserves_t_start():
    """Resampling should preserve t_start when the parent has one."""
    sampling_frequency = 30000
    t_start = 100.5
    traces = np.random.randn(sampling_frequency * 2, 2).astype(np.float32)
    parent_rec = NumpyRecording(traces, sampling_frequency)
    parent_rec.segments[0].t_start = t_start

    resampled = resample(parent_rec, 500)
    assert resampled.segments[0].t_start == t_start
    assert not resampled.has_time_vector()
    assert np.isclose(resampled.get_times()[0], t_start)


def test_resample_does_not_mutate_parent():
    """Resampling should not modify the parent recording's time_vector."""
    sampling_frequency = 30000
    n_samples = sampling_frequency * 2
    traces = np.random.randn(n_samples, 2).astype(np.float32)
    parent_rec = NumpyRecording(traces, sampling_frequency)
    time_vector = np.arange(n_samples, dtype="float64") / sampling_frequency + 50.0
    parent_rec.set_times(time_vector)

    assert parent_rec.has_time_vector()
    resample(parent_rec, 500)
    assert parent_rec.has_time_vector(), "Parent time_vector was mutated by resample!"
    np.testing.assert_array_equal(parent_rec.get_times(), time_vector)


def test_resample_preserves_time_vector_integer_ratio():
    """Resampling with integer ratio should slice the parent time_vector,
    preserving gaps when gap_tolerance_ms is provided."""
    sampling_frequency = 30000
    resample_rate = 500
    n_samples = sampling_frequency * 2
    traces = np.random.randn(n_samples, 2).astype(np.float32)
    parent_rec = NumpyRecording(traces, sampling_frequency)

    # Create a time_vector with a gap (simulating artifact removal)
    time_vector = np.arange(n_samples, dtype="float64") / sampling_frequency
    # Insert a 5-second gap at the midpoint
    midpoint = n_samples // 2
    time_vector[midpoint:] += 5.0
    parent_rec.set_times(time_vector)

    resampled = resample(parent_rec, resample_rate, gap_tolerance_ms=1.0)

    assert resampled.has_time_vector()
    resampled_times = resampled.get_times()
    n_out = resampled.get_num_samples()

    # Output length should be consistent
    assert len(resampled_times) == n_out

    # The gap should be preserved: check that the jump exists in the resampled times
    diffs = np.diff(resampled_times)
    normal_dt = 1.0 / resample_rate
    gap_indices = np.where(diffs > normal_dt * 2)[0]
    assert len(gap_indices) == 1, "The gap should appear exactly once in resampled times"
    assert np.isclose(diffs[gap_indices[0]], normal_dt + 5.0, atol=normal_dt)

    # Start time should match
    assert np.isclose(resampled_times[0], time_vector[0])


def test_resample_preserves_time_vector_non_integer_ratio():
    """Resampling with non-integer ratio should interpolate the time_vector."""
    sampling_frequency = 30000
    resample_rate = 700  # 30000 / 700 is not integer
    n_samples = sampling_frequency * 2
    traces = np.random.randn(n_samples, 2).astype(np.float32)
    parent_rec = NumpyRecording(traces, sampling_frequency)

    time_vector = np.arange(n_samples, dtype="float64") / sampling_frequency + 10.0
    parent_rec.set_times(time_vector)

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        resampled = resample(parent_rec, resample_rate)
        assert any("non-integer ratio" in str(warning.message).lower() for warning in w)

    assert resampled.has_time_vector()
    resampled_times = resampled.get_times()
    assert len(resampled_times) == resampled.get_num_samples()
    assert np.isclose(resampled_times[0], 10.0, atol=1.0 / sampling_frequency)


def test_resample_errors_on_gaps_by_default():
    """With gap_tolerance_ms=None (default), a gapped time vector should raise ValueError."""
    rec, _, _ = _make_gapped_recording()
    with pytest.raises(ValueError, match="timestamp gap"):
        resample(rec, 500)


def test_resample_preserves_gaps_non_integer_ratio():
    """Non-integer ratio with gap_tolerance_ms should preserve the gap in the output time_vector."""
    sampling_frequency = 30000
    resample_rate = 700  # non-integer ratio
    gap_s = 5.0
    rec, n1, n2 = _make_gapped_recording(sampling_frequency=sampling_frequency, gap_s=gap_s)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        resampled = resample(rec, resample_rate, gap_tolerance_ms=1.0)

    assert resampled.has_time_vector()
    resampled_times = resampled.get_times()
    assert len(resampled_times) == resampled.get_num_samples()

    # The gap should be preserved
    diffs = np.diff(resampled_times)
    normal_dt = 1.0 / resample_rate
    gap_indices = np.where(diffs > normal_dt * 2)[0]
    assert len(gap_indices) == 1, f"Expected 1 gap, found {len(gap_indices)}"

    # Gap size should be approximately gap_s (plus one normal dt)
    assert np.isclose(diffs[gap_indices[0]], gap_s + normal_dt, atol=2 * normal_dt)

    # No timestamps should fall inside the gap
    parent_tv = rec.get_times()
    gap_start_t = parent_tv[n1 - 1]
    gap_end_t = parent_tv[n1]
    in_gap = (resampled_times > gap_start_t + normal_dt) & (resampled_times < gap_end_t - normal_dt)
    assert not np.any(in_gap), "Output timestamps fall inside the gap"


def test_resample_traces_across_gap():
    """Section-wise resampling should match individually resampled sections.

    Build a gapped recording, resample it with gap_tolerance_ms, and verify
    that each section's output matches what you'd get by resampling that
    section alone (without the gap).  This confirms that _get_traces_gapped
    does not apply FFT processing across gap boundaries.
    """
    sampling_frequency = 30000
    resample_rate = 700  # non-integer ratio
    sec_duration = 2.0
    gap_s = 5.0

    n1 = int(sec_duration * sampling_frequency)
    n2 = int(sec_duration * sampling_frequency)

    # Build random traces (more realistic than a sinusoid)
    rng = np.random.default_rng(42)
    traces1 = rng.standard_normal((n1, 2)).astype(np.float32)
    traces2 = rng.standard_normal((n2, 2)).astype(np.float32)
    traces = np.concatenate([traces1, traces2], axis=0)

    t1 = np.arange(n1, dtype="float64") / sampling_frequency
    t2 = np.arange(n2, dtype="float64") / sampling_frequency + sec_duration + gap_s
    tv = np.concatenate([t1, t2])

    rec = NumpyRecording(traces, sampling_frequency)
    rec.set_times(tv)

    # Resample the gapped recording
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        resampled = resample(rec, resample_rate, gap_tolerance_ms=1.0)

    resampled_traces = resampled.get_traces()
    n_out_1 = int(resampled.segments[0]._sec_n_out[0])
    n_out_2 = int(resampled.segments[0]._sec_n_out[1])
    assert resampled_traces.shape[0] == n_out_1 + n_out_2

    # Resample each section independently (no gap in these recordings)
    rec1 = NumpyRecording(traces1, sampling_frequency)
    rec1.set_times(t1)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        resampled1 = resample(rec1, resample_rate, gap_tolerance_ms=1.0)
    ref_traces1 = resampled1.get_traces()

    rec2 = NumpyRecording(traces2, sampling_frequency)
    rec2.set_times(t2)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        resampled2 = resample(rec2, resample_rate, gap_tolerance_ms=1.0)
    ref_traces2 = resampled2.get_traces()

    # Section 1 should match the independently resampled section 1
    gapped_s1 = resampled_traces[:n_out_1]
    assert gapped_s1.shape == ref_traces1.shape, (
        f"Section 1 shape mismatch: {gapped_s1.shape} vs {ref_traces1.shape}"
    )
    np.testing.assert_allclose(gapped_s1, ref_traces1, rtol=1e-5, atol=1e-5)

    # Section 2 should match the independently resampled section 2
    gapped_s2 = resampled_traces[n_out_1 : n_out_1 + n_out_2]
    assert gapped_s2.shape == ref_traces2.shape, (
        f"Section 2 shape mismatch: {gapped_s2.shape} vs {ref_traces2.shape}"
    )
    np.testing.assert_allclose(gapped_s2, ref_traces2, rtol=1e-5, atol=1e-5)


def test_resample_gapped_chunked_consistency():
    """Chunked .save() should match non-chunked for gapped recordings."""
    sampling_frequency = 30000
    resample_rate = 700
    rec, _, _ = _make_gapped_recording(sampling_frequency=sampling_frequency, sec1_duration=2.0, sec2_duration=2.0)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        resampled = resample(rec, resample_rate, gap_tolerance_ms=1.0)

    traces_full = resampled.get_traces()
    chunk_size = resample_rate  # 1 second chunks
    saved = resampled.save(format="memory", chunk_size=chunk_size, n_jobs=1, progress_bar=False)
    traces_chunked = saved.get_traces()

    assert traces_full.shape == traces_chunked.shape
    # Interior samples should match closely (edges may have small differences)
    sl = slice(chunk_size, -chunk_size)
    rms = np.sqrt(np.mean(traces_full[sl] ** 2))
    if rms > 0:
        error = np.sqrt(np.mean((traces_full[sl] - traces_chunked[sl]) ** 2))
        assert error / rms < 0.05, f"Chunked vs full RMS error ratio: {error / rms:.4f}"


def test_resample_no_gap_unchanged_behavior():
    """Uniform time_vector without gaps should produce identical results with or without gap_tolerance_ms."""
    sampling_frequency = 30000
    resample_rate = 700
    n_samples = sampling_frequency * 2
    traces = np.random.randn(n_samples, 1).astype(np.float32)
    rec = NumpyRecording(traces, sampling_frequency)

    tv = np.arange(n_samples, dtype="float64") / sampling_frequency + 100.0
    rec.set_times(tv)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        # Without gap_tolerance_ms (no gaps, so no error)
        resampled_default = resample(rec, resample_rate)
        # With gap_tolerance_ms (no gaps to split on, should be identical)
        resampled_tolerant = resample(rec, resample_rate, gap_tolerance_ms=1.0)

    np.testing.assert_array_equal(resampled_default.get_times(), resampled_tolerant.get_times())
    np.testing.assert_array_equal(resampled_default.get_traces(), resampled_tolerant.get_traces())


def test_resample_multiple_gaps():
    """Recording with multiple gaps should produce the correct number of sections."""
    sampling_frequency = 30000
    resample_rate = 700
    n_per_sec = int(0.5 * sampling_frequency)  # 0.5s per section
    n_sections = 4
    n_total = n_per_sec * n_sections
    traces = np.random.randn(n_total, 1).astype(np.float32)
    rec = NumpyRecording(traces, sampling_frequency)

    # Create time_vector with 3 gaps (between 4 sections)
    tv = np.arange(n_total, dtype="float64") / sampling_frequency
    for i in range(1, n_sections):
        tv[i * n_per_sec :] += (i * 10.0)  # gaps of 10s, 20s, 30s cumulative offsets
    rec.set_times(tv)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        resampled = resample(rec, resample_rate, gap_tolerance_ms=1.0)

    resampled_times = resampled.get_times()
    diffs = np.diff(resampled_times)
    normal_dt = 1.0 / resample_rate

    # Should detect 3 gaps
    gap_indices = np.where(diffs > normal_dt * 2)[0]
    assert len(gap_indices) == 3, f"Expected 3 gaps, found {len(gap_indices)}"

    # Traces and times should match in length
    assert resampled.get_traces().shape[0] == len(resampled_times)


def test_resample_gap_tolerance_filtering():
    """Gaps smaller than gap_tolerance_ms should be treated as continuous."""
    sampling_frequency = 30000
    resample_rate = 500  # integer ratio for simplicity

    n1 = int(1.0 * sampling_frequency)
    n2 = int(1.0 * sampling_frequency)
    n3 = int(1.0 * sampling_frequency)
    n_total = n1 + n2 + n3
    traces = np.random.randn(n_total, 1).astype(np.float32)
    rec = NumpyRecording(traces, sampling_frequency)

    tv = np.arange(n_total, dtype="float64") / sampling_frequency
    # Small gap (5ms = 0.005s) after section 1 — detectable but below 50ms tolerance
    tv[n1:] += 0.005
    # Large gap (100ms = 0.1s) after section 2
    tv[n1 + n2 :] += 0.1
    rec.set_times(tv)

    # With tolerance of 50ms: only the 100ms gap triggers a section split
    resampled = resample(rec, resample_rate, gap_tolerance_ms=50.0)
    seg = resampled.segments[0]
    assert seg._has_gaps, "Should detect at least one gap"
    n_sections = len(seg._sec_n_out)
    assert n_sections == 2, f"Expected 2 sections (split at 100ms gap), found {n_sections}"

    # With tolerance of 1.0ms: both gaps trigger section splits
    # (5ms > 1ms, 100ms > 1ms)
    resampled_strict = resample(rec, resample_rate, gap_tolerance_ms=1.0)
    seg_strict = resampled_strict.segments[0]
    n_sections_strict = len(seg_strict._sec_n_out)
    assert n_sections_strict == 3, f"Expected 3 sections with 1ms tolerance, found {n_sections_strict}"


if __name__ == "__main__":
    test_resample_freq_domain()
    test_resample_by_chunks()
    test_resample_preserves_t_start()
    test_resample_does_not_mutate_parent()
    test_resample_preserves_time_vector_integer_ratio()
    test_resample_preserves_time_vector_non_integer_ratio()
    test_resample_errors_on_gaps_by_default()
    test_resample_preserves_gaps_non_integer_ratio()
    test_resample_traces_across_gap()
    test_resample_gapped_chunked_consistency()
    test_resample_no_gap_unchanged_behavior()
    test_resample_multiple_gaps()
    test_resample_gap_tolerance_filtering()
