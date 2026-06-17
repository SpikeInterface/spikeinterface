import pytest


from spikeinterface import NumpyRecording
from spikeinterface.core import generate_recording, load
from spikeinterface.preprocessing.decimate import DecimateRecording, decimate, get_balanced_decimation_factors
from spikeinterface.preprocessing.tests.test_resample import create_sinusoidal_traces
import numpy as np


@pytest.mark.parametrize("num_segments", [1, 2])
@pytest.mark.parametrize("decimation_offset", [0, 1, 5, 21, 101])
@pytest.mark.parametrize("decimation_factor", [1, 7, 50])
def test_decimate(num_segments, decimation_offset, decimation_factor):
    segment_num_samps = [20000, 40000]
    rec = NumpyRecording([np.arange(2 * N).reshape(N, 2) for N in segment_num_samps], 1)

    parent_traces = [rec.get_traces(i) for i in range(num_segments)]

    if decimation_offset >= min(segment_num_samps) or decimation_offset >= decimation_factor:
        with pytest.raises(ValueError):
            decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
        return

    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
    decimated_parent_traces = [parent_traces[i][decimation_offset::decimation_factor] for i in range(num_segments)]

    for start_frame in [0, 1, 5, None, 1000]:
        for end_frame in [0, 1, 5, None, 1000]:
            if start_frame is None:
                start_frame = max(decimated_rec.get_num_samples(i) for i in range(num_segments))
            if end_frame is None:
                end_frame = max(decimated_rec.get_num_samples(i) for i in range(num_segments))

            for i in range(num_segments):
                assert decimated_rec.get_num_samples(i) == decimated_parent_traces[i].shape[0]
                assert np.all(
                    decimated_rec.get_traces(i, start_frame, end_frame)
                    == decimated_parent_traces[i][start_frame:end_frame]
                )

    for i in range(num_segments):
        assert decimated_rec.get_num_samples(i) == decimated_parent_traces[i].shape[0]
        assert np.all(
            decimated_rec.get_traces(i, start_frame, end_frame) == decimated_parent_traces[i][start_frame:end_frame]
        )


@pytest.mark.parametrize("antialias", [False, True])
def test_decimate_with_times(antialias):
    rec = generate_recording(durations=[5, 10])

    # test with times
    times = [rec.get_times(0) + 10, rec.get_times(1) + 20]
    for i, t in enumerate(times):
        rec.set_times(t, i)

    decimation_factor = 2
    decimation_offset = 1
    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset, antialias=antialias)

    for segment_index in range(rec.get_num_segments()):
        assert np.allclose(
            decimated_rec.get_times(segment_index),
            rec.get_times(segment_index)[decimation_offset::decimation_factor],
        )

    # test with t_start
    rec = generate_recording(durations=[5, 10])
    t_starts = [10, 20]
    for t_start, rec_segment in zip(t_starts, rec.segments):
        rec_segment.t_start = t_start
    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset, antialias=antialias)
    for segment_index in range(rec.get_num_segments()):
        assert np.allclose(
            decimated_rec.get_times(segment_index),
            rec.get_times(segment_index)[decimation_offset::decimation_factor],
        )


@pytest.mark.parametrize(
    "decimation_factor, expected",
    [
        (1, [1]),
        (7, [7]),
        (13, [13]),
        (48, [8, 6]),
        (50, [10, 5]),
        (60, [10, 6]),
        (100, [10, 10]),
        (17, [17]),  # prime > 13: cannot be split
        (23, [23]),  # prime > 13: cannot be split
    ],
)
def test_balanced_decimation_factors(decimation_factor, expected):
    factors = get_balanced_decimation_factors(decimation_factor)
    assert factors == expected
    # The product of the sub-factors always reconstructs the requested factor.
    assert int(np.prod(factors)) == decimation_factor
    # Every pass is <= 13 unless the factor is an unsplittable prime > 13.
    if len(factors) > 1:
        assert all(f <= 13 for f in factors)


@pytest.mark.parametrize("decimation_factor", [6, 10, 48])
def test_decimate_antialias_by_chunks(decimation_factor):
    # Mirror test_resample_by_chunks: chunked reads must match a full read once the
    # anti-aliasing margins are accounted for. Factor 48 exercises the internal multi-pass.
    sampling_frequency = int(3e4)
    duration = 30
    traces, _ = create_sinusoidal_traces(sampling_frequency, duration, freqs_n=10, max_freq=1000, dtype=np.float32)
    parent_rec = NumpyRecording(traces, sampling_frequency)
    rms = np.sqrt(np.mean(parent_rec.get_traces() ** 2))
    decimated_rate = sampling_frequency / decimation_factor

    for margin_ms in [100, 1000]:
        rec2 = DecimateRecording(parent_rec, decimation_factor, antialias=True, margin_ms=margin_ms)
        chunk_size = int(decimated_rate * 2)  # ~2 seconds of the decimated signal
        rec3 = rec2.save(format="memory", chunk_size=chunk_size, n_jobs=1, progress_bar=False)

        traces2 = rec2.get_traces()
        traces3 = rec3.get_traces()

        # Drop the first and last chunk before comparing (as in test_resample_by_chunks).
        sl = slice(chunk_size, -chunk_size)
        error_mean = np.sqrt(np.mean((traces2[sl] - traces3[sl]) ** 2))
        error_max = np.sqrt(np.max((traces2[sl] - traces3[sl]) ** 2))

        assert error_mean / rms < 0.01
        assert error_max / rms < 0.05


@pytest.mark.parametrize("decimation_factor", [6, 10])
@pytest.mark.parametrize("decimation_offset", [0, 1, 5])
def test_decimate_antialias_with_offset(decimation_factor, decimation_offset):
    sampling_frequency = 30000
    # max_freq below every tested Nyquist, so anti-aliasing barely changes the signal.
    traces, _ = create_sinusoidal_traces(sampling_frequency, duration=5, freqs_n=6, max_freq=500, dtype=np.float32)
    parent_rec = NumpyRecording(traces, sampling_frequency)

    dec_aa = DecimateRecording(
        parent_rec, decimation_factor, decimation_offset=decimation_offset, antialias=True, dtype="float32"
    )
    dec_plain = DecimateRecording(
        parent_rec, decimation_factor, decimation_offset=decimation_offset, antialias=False, dtype="float32"
    )

    # The anti-aliasing path returns the same number of samples as plain slicing.
    parent_n = parent_rec.get_num_samples()
    expected_n = int(np.ceil((parent_n - decimation_offset) / decimation_factor))
    assert dec_aa.get_num_samples() == expected_n
    assert dec_aa.get_num_samples() == dec_plain.get_num_samples()

    # With only sub-Nyquist content, anti-aliased and plain-sliced traces stay aligned.
    corr = np.corrcoef(dec_aa.get_traces().ravel(), dec_plain.get_traces().ravel())[0, 1]
    assert corr > 0.95


def test_decimate_antialias_multipass():
    sampling_frequency = 30000
    decimation_factor = 48
    traces, _ = create_sinusoidal_traces(sampling_frequency, duration=10, freqs_n=8, max_freq=200, dtype=np.float32)
    parent_rec = NumpyRecording(traces, sampling_frequency)

    dec = decimate(parent_rec, decimation_factor, antialias=True)

    # Multi-pass happens internally: a single DecimateRecording carries the full factor.
    assert isinstance(dec, DecimateRecording)
    assert dec._kwargs["decimation_factor"] == decimation_factor

    segment = dec.segments[0]
    assert int(np.prod(segment._antialias_factors)) == decimation_factor
    assert all(f <= 13 for f in segment._antialias_factors)

    parent_n = parent_rec.get_num_samples()
    assert dec.get_num_samples() == int(np.ceil(parent_n / decimation_factor))

    # Provenance round-trips and reproduces the traces.
    dec_loaded = load(dec.to_dict())
    np.testing.assert_allclose(dec_loaded.get_traces(), dec.get_traces())


def test_decimate_antialias_large_prime_warns():
    rec = generate_recording(durations=[2.0], num_channels=2)
    with pytest.warns(UserWarning, match="prime factor > 13"):
        dec = DecimateRecording(rec, 17, antialias=True)
    # The unsplittable factor falls back to a single pass.
    assert dec.segments[0]._antialias_factors == [17]


if __name__ == "__main__":
    test_decimate()
