import numpy as np
from spikeinterface import NumpyRecording

from spikeinterface.preprocessing import phase_shift
from spikeinterface.preprocessing.phase_shift import apply_frequency_shift


def create_shifted_channel(inter_sample_shift=None):
    duration = 5.0
    sr_h = 10000.0
    times_h = np.arange(0, duration, 1 / sr_h)
    freq1 = 2.5
    freq2 = 8.5
    sig_h = np.sin(2 * np.pi * freq1 * times_h) + np.sin(2 * np.pi * freq2 * times_h)
    # ~ noise = np.random.randn(sig_h.size)
    # ~ sig_h += noise * 0.02

    ratio = 10
    sr = sr_h / ratio
    if inter_sample_shift is None:
        inter_sample_shift = [0.0, 0.4]
    channels = [sig_h[int(round(shift * ratio)) :: ratio] for shift in inter_sample_shift]
    min_length = min(channel.size for channel in channels)
    traces = np.stack([channel[:min_length] for channel in channels], axis=1) * 1000
    return traces, sr, inter_sample_shift


def test_phase_shift():
    traces, sampling_frequency, inter_sample_shift = create_shifted_channel()
    # traces = (traces * 1000).astype('int16')

    # ~ print(sampling_frequency)

    for dtype in ("float64", "float32", "int16"):
        rec = NumpyRecording([traces.astype(dtype)], sampling_frequency)
        rec.set_property("inter_sample_shift", inter_sample_shift)
        original_traces = rec.get_traces(end_frame=10)

        for margin_ms in (10.0, 30.0, 40.0):
            for chunk_size in (100, 500, 1000, 2000):
                rec2 = phase_shift(rec, margin_ms=margin_ms)
                assert rec2.dtype == rec.dtype

                # save by chunk rec3 is the cached version
                rec3 = rec2.save(format="memory", chunk_size=chunk_size, n_jobs=1, progress_bar=False)

                traces2 = rec2.get_traces()
                assert traces2.dtype == original_traces.dtype
                traces3 = rec3.get_traces()
                assert traces3.dtype == original_traces.dtype

                traces_slice = rec3.get_traces(channel_ids=[rec3.channel_ids[0]])
                assert traces_slice.shape[1] == 1

                # error between full and chunked
                error_mean = np.sqrt(np.mean((traces2 - traces3) ** 2))
                error_max = np.sqrt(np.max((traces2 - traces3) ** 2))
                rms = np.sqrt(np.mean(traces**2))

                # this will never be possible:
                #      assert np.allclose(traces2, traces3)
                # so we check that the diff between chunk processing and not chunked is small
                # ~ print()
                # ~ print(dtype, margin_ms, chunk_size)
                # ~ print(error_mean, rms, error_mean / rms)
                # ~ print(error_max, rms, error_max / rms)
                assert error_mean / rms < 0.001
                assert error_max / rms < 0.02

                # ~ import matplotlib.pyplot as plt
                # ~ fig, axs = plt.subplots(nrows=3, sharex=True)
                # ~ ax = axs[0]
                # ~ ax.set_title(f'margin_ms{margin_ms} chunk_size{chunk_size} {error_max/rms:.6f} {error_mean/rms:.6f}')
                # ~ ax.plot(traces[:, 0], color='r', label='no delay')
                # ~ ax.plot(traces[:, 1], color='b', label='delay')
                # ~ ax.plot(traces2[:, 1], color='c', ls='--', label='shift no chunk')
                # ~ ax.plot(traces3[:, 1], color='g', ls='--', label='shift no chunked')
                # ~ ax = axs[1]
                # ~ ax.plot(traces2[:, 1] - traces3[:, 1], color='k')
                # ~ ax = axs[2]
                # ~ ax.plot(traces2[:, 1] - traces[:, 0], color='c')
                # ~ ax.plot(traces3[:, 1] - traces[:, 0], color='g')
                # ~ plt.show()

    # ~ import matplotlib.pyplot as plt
    # ~ import spikeinterface.full as si
    # ~ si.plot_traces(rec, segment_index=0, time_range=[0, 10])
    # ~ si.plot_traces(rec2, segment_index=0, time_range=[0, 10])
    # ~ si.plot_traces(rec3, segment_index=0, time_range=[0, 10])
    # ~ plt.show()


def test_phase_shift_repeated_delays():
    inter_sample_shift = np.tile(np.arange(8) / 10, 8)
    traces, sampling_frequency, _ = create_shifted_channel(inter_sample_shift)
    recording = NumpyRecording([traces], sampling_frequency)
    recording.set_property("inter_sample_shift", inter_sample_shift)

    recording2 = phase_shift(recording, margin_ms=40.0)
    traces2 = recording2.get_traces()
    recording3 = recording2.save(format="memory", chunk_size=1000, n_jobs=1, progress_bar=False)
    traces3 = recording3.get_traces()

    rms = np.sqrt(np.mean(traces**2))
    error_mean = np.sqrt(np.mean((traces2 - traces3) ** 2))
    error_max = np.sqrt(np.max((traces2 - traces3) ** 2))
    assert error_mean / rms < 0.001
    assert error_max / rms < 0.02

    # Reading from recording2 calls apply_frequency_shift again. These non-contiguous blocks retain eight
    # delays across 32 channels, so the subset also enters the gather path at the exact 4x boundary.
    subset_indices = np.r_[0:16, 32:48]
    subset_ids = recording2.channel_ids[subset_indices]
    traces_slice = recording2.get_traces(channel_ids=subset_ids)
    np.testing.assert_allclose(traces_slice, traces2[:, subset_indices], rtol=1e-12, atol=1e-12)


def test_apply_frequency_shift_repeated_and_unique_shifts():
    rng = np.random.default_rng(0)
    boundary_shifts = np.tile(np.arange(8) / 8, 4)
    skewed_shifts = np.concatenate([np.full(25, 0.1), np.arange(1, 8) / 9])
    rng.shuffle(skewed_shifts)
    _, skewed_counts = np.unique(skewed_shifts, return_counts=True)
    assert np.unique(boundary_shifts).size * 4 == boundary_shifts.size
    assert skewed_counts.min() == 1 and skewed_counts.max() == 25

    cases = []
    # 100/101 sit below the 128-sample fast-path eligibility floor (fallback path); 128/129 sit at/above it
    # (gather path), so both branches are covered for repeated, unique, exact-4x, and skewed shifts.
    for num_samples in (100, 101, 128, 129):
        signal = rng.standard_normal((num_samples, 32))
        shift_cases = (np.tile(np.arange(4) / 4, 8), np.arange(32) / 32, boundary_shifts, skewed_shifts)
        for shift_samples in shift_cases:
            frequencies = 2 * np.pi * np.fft.rfftfreq(signal.shape[0])[:, np.newaxis]
            expected = np.fft.irfft(
                np.fft.rfft(signal, axis=0) * np.exp(-1j * frequencies * shift_samples[np.newaxis, :]),
                n=signal.shape[0],
                axis=0,
            )
            cases.append((signal, shift_samples, expected))

    for signal, shift_samples, expected in cases:
        actual = apply_frequency_shift(signal.copy(), shift_samples)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_apply_frequency_shift_float32_input_matches_float64_oracle():
    rng = np.random.default_rng(0)
    # 128 samples clears the fast-path signal-length floor so repeated_shifts below exercises the
    # repeated-shift branch this oracle targets, not the fallback.
    signal32 = rng.standard_normal((128, 32)).astype(np.float32)
    repeated_shifts = np.tile(np.arange(4) / 4, 8)
    unique_shifts = np.arange(32) / 32
    frequencies = 2 * np.pi * np.fft.rfftfreq(signal32.shape[0])[:, np.newaxis]

    for shift_samples in (repeated_shifts, unique_shifts):
        # Independent oracle: upcast the signal to float64 before the FFT, rather than after, as the
        # repeated-shift branch does. The two are not bit-identical (the float32 FFT already rounds at
        # float32 precision) but must agree well past that noise floor.
        expected = np.fft.irfft(
            np.fft.rfft(signal32.astype(np.float64), axis=0) * np.exp(-1j * frequencies * shift_samples[np.newaxis, :]),
            n=signal32.shape[0],
            axis=0,
        )
        actual = apply_frequency_shift(signal32.copy(), shift_samples)
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_apply_frequency_shift_output_dtype_independent_of_shift_dtype():
    rng = np.random.default_rng(0)
    # 128 samples clears the fast-path signal-length floor so repeated_shifts below exercises the
    # repeated-shift branch this test targets, not the fallback.
    signal = rng.standard_normal((128, 32))

    repeated_shifts = np.tile(np.array([0.13, 0.27, 0.41, 0.59]), 8)
    unique_shifts = np.linspace(0.013, 0.917, 32)
    for shift_samples in (repeated_shifts, unique_shifts):
        reference = apply_frequency_shift(signal.copy(), shift_samples)
        widened = apply_frequency_shift(signal.copy(), shift_samples.astype(np.longdouble))
        assert widened.dtype == reference.dtype == np.float64
        np.testing.assert_allclose(widened, reference, rtol=1e-12, atol=1e-12)


def test_apply_frequency_shift_repeated_path_rounds_after_multiplying_not_before():
    # Parse directly into longdouble: casting up a float64 cannot recover precision it never had.
    wide_shift = np.longdouble("1000000000000.0001")
    # 128 samples clears the fast-path signal-length floor so this exercises the repeated-shift branch.
    num_samples, num_channels = 128, 32
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((num_samples, num_channels))
    shift_samples = np.full(num_channels, wide_shift, dtype=np.longdouble)

    # Independent oracle: multiply at the shift's native precision, then round the angle to float64
    # at the same point as the original vectorized path's float64 output buffer.
    angular_frequencies = 2 * np.pi * np.fft.rfftfreq(num_samples)
    angles = (angular_frequencies[:, np.newaxis] * shift_samples[np.newaxis, :]).astype(np.float64)
    rotations = np.exp(-1j * angles)
    expected = np.fft.irfft(np.fft.rfft(signal, axis=0) * rotations, n=num_samples, axis=0)

    actual = apply_frequency_shift(signal.copy(), shift_samples)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_phase_shift()
