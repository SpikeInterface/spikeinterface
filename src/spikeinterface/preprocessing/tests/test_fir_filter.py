import numpy as np
import pytest

from spikeinterface import NumpyRecording, load
from spikeinterface.core.testing import check_chunk_invariance
from spikeinterface.preprocessing import design_fir_bandpass, design_fir_highpass, fir_filter

FS = 30_000.0


@pytest.fixture(scope="module")
def recording():
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((20_000, 6)) * 50).astype(np.float32)
    return NumpyRecording(traces, FS)


def expected_output(traces, taps, center):
    """``sum_j taps[j] * x[n + center - j]`` with zeros beyond the edges, by direct convolution."""
    full = np.stack([np.convolve(traces[:, c].astype(np.float64), taps) for c in range(traces.shape[1])], axis=1)
    return full[center : center + traces.shape[0]]


def bandpass_taps():
    return design_fir_bandpass(300.0, 6000.0, FS)


@pytest.mark.parametrize("center", [None, 0, 7])
def test_fir_filter_matches_direct_convolution(recording, center):
    taps = np.random.default_rng(1).standard_normal(31)
    traces = recording.get_traces()
    out = fir_filter(recording, taps, center=center).get_traces()
    expected = expected_output(traces, taps, 15 if center is None else center)
    np.testing.assert_allclose(out, expected, atol=1e-5 * np.abs(expected).max())


def test_fir_filter_causal(recording):
    # With center=0 an impulse at sample n moves only samples n and later.
    traces = np.zeros((1000, 1), dtype=np.float32)
    traces[500] = 1.0
    taps = np.arange(1.0, 6.0)
    out = fir_filter(NumpyRecording(traces, FS), taps, center=0).get_traces()[:, 0]
    np.testing.assert_allclose(out[:500], 0.0, atol=1e-6)
    np.testing.assert_allclose(out[500:505], taps, rtol=1e-6)


def test_fir_filter_chunk_invariance(recording):
    rec = fir_filter(recording, bandpass_taps())
    scale = np.abs(rec.get_traces()).max()
    for chunk_size in (997, 3000, 7001):
        check_chunk_invariance(rec, chunk_size, atol=1e-5 * scale)


def test_fir_filter_per_channel_kernels(recording):
    rng = np.random.default_rng(2)
    kernels = rng.standard_normal((3, 21))
    groups = np.array([0, 1, 2, 0, 1, 2])
    traces = recording.get_traces()
    out = fir_filter(recording, kernels, channel_groups=groups).get_traces()
    for channel, group in enumerate(groups):
        expected = expected_output(traces[:, [channel]], kernels[group], 10)
        np.testing.assert_allclose(out[:, [channel]], expected, atol=1e-5 * np.abs(expected).max())
    # One row per channel needs no channel_groups.
    one_per_channel = rng.standard_normal((6, 21))
    out = fir_filter(recording, one_per_channel).get_traces()
    expected = expected_output(traces[:, [4]], one_per_channel[4], 10)
    np.testing.assert_allclose(out[:, [4]], expected, atol=1e-5 * np.abs(expected).max())


def test_fir_filter_channel_blocks(recording, monkeypatch):
    import sys

    # Blocks of 4 split the 6 channels unevenly and across kernel groups.
    monkeypatch.setattr(
        sys.modules["spikeinterface.preprocessing.fir_filter"], "_channels_per_block", lambda *a, **k: 4
    )
    kernels = np.random.default_rng(3).standard_normal((2, 21))
    groups = np.array([0, 1, 1, 0, 1, 0])
    traces = recording.get_traces()
    out = fir_filter(recording, kernels, channel_groups=groups).get_traces()
    expected = np.concatenate([expected_output(traces[:, [c]], kernels[g], 10) for c, g in enumerate(groups)], axis=1)
    np.testing.assert_allclose(out, expected, atol=1e-5 * np.abs(expected).max())


def test_fir_filter_channel_subset(recording):
    rng = np.random.default_rng(3)
    kernels = rng.standard_normal((6, 41))
    rec = fir_filter(recording, kernels)
    whole = rec.get_traces()
    ids = recording.channel_ids[[4, 1]]
    subset = rec.get_traces(start_frame=1234, end_frame=5678, channel_ids=ids)
    np.testing.assert_allclose(subset, whole[1234:5678, [4, 1]], atol=1e-5 * np.abs(whole).max())


def test_fir_filter_integer_output_rounds(recording):
    int_rec = NumpyRecording(np.rint(recording.get_traces()).astype(np.int16), FS)
    taps = bandpass_taps()
    out = fir_filter(int_rec, taps).get_traces()
    assert out.dtype == np.int16
    as_float = fir_filter(int_rec, taps, dtype="float32").get_traces()
    np.testing.assert_array_equal(out, np.rint(as_float).astype(np.int16))


def test_fir_filter_reflect_padding():
    import scipy.signal

    # A constant signal reflected at the edges stays constant through a unit-gain lowpass.
    traces = np.full((2000, 1), 10.0, dtype=np.float32)
    taps = scipy.signal.firwin(101, 1000.0, fs=FS)
    rec = NumpyRecording(traces, FS)
    np.testing.assert_allclose(fir_filter(rec, taps, add_reflect_padding=True).get_traces(), traces, rtol=1e-4)
    assert abs(fir_filter(rec, taps).get_traces()[0, 0] - 10.0) > 1.0


def test_fir_filter_offsets_scale_by_dc_gain(recording):
    import scipy.signal

    rec = NumpyRecording(recording.get_traces(), FS)
    rec.set_channel_gains(2.0)
    rec.set_channel_offsets(5.0)
    taps = bandpass_taps()
    np.testing.assert_allclose(fir_filter(rec, taps).get_channel_offsets(), 5.0 * taps.sum())
    lowpass = scipy.signal.firwin(101, 1000.0, fs=FS)
    np.testing.assert_allclose(fir_filter(rec, lowpass).get_channel_offsets(), 5.0 * lowpass.sum())


def test_fir_filter_save_load(tmp_path):
    from spikeinterface.core import generate_recording

    parent = generate_recording(num_channels=4, durations=[1.0], sampling_frequency=FS, seed=0)
    rec = fir_filter(parent, np.random.default_rng(4).standard_normal((2, 15)), channel_groups=[0, 1, 0, 1])
    rec.dump_to_json(tmp_path / "rec.json")
    np.testing.assert_array_equal(load(tmp_path / "rec.json").get_traces(), rec.get_traces())
    saved = rec.save(folder=tmp_path / "saved")
    np.testing.assert_array_equal(saved.get_traces(), rec.get_traces())


def test_fir_filter_invalid_arguments(recording):
    with pytest.raises(ValueError, match="channel_groups"):
        fir_filter(recording, np.ones((3, 5)))
    with pytest.raises(ValueError, match="channel_groups"):
        fir_filter(recording, np.ones((3, 5)), channel_groups=[0, 1, 2, 3, 0, 1])
    with pytest.raises(ValueError, match="center"):
        fir_filter(recording, np.ones(5), center=5)


def response_db(taps, freqs):
    import scipy.signal

    return 20 * np.log10(np.abs(scipy.signal.freqz(taps, worN=np.asarray(freqs, dtype=float), fs=FS)[1]))


@pytest.mark.parametrize(
    "freq_min, freq_max, transition_low_hz, transition_high_hz, stopband_db",
    [(300.0, 6000.0, 175.0, 2000.0, 45.0), (300.0, 6000.0, 400.0, 2000.0, 60.0), (500.0, 3000.0, 600.0, 300.0, 40.0)],
)
def test_design_fir_bandpass_meets_spec(freq_min, freq_max, transition_low_hz, transition_high_hz, stopband_db):
    taps = design_fir_bandpass(freq_min, freq_max, FS, transition_low_hz, transition_high_hz, stopband_db)
    assert taps.size % 2 == 1
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-15)
    np.testing.assert_allclose(response_db(taps, [freq_min, freq_max]), -6.02, atol=0.1)
    stop_low = np.linspace(0.0, freq_min - transition_low_hz / 2, 2000)
    stop_high = np.linspace(freq_max + transition_high_hz / 2, FS / 2, 2000)
    assert response_db(taps, np.r_[stop_low, stop_high]).max() <= -stopband_db
    passband = np.linspace(freq_min + transition_low_hz / 2, freq_max - transition_high_hz / 2, 2000)
    delta = 10 ** (-stopband_db / 20)
    gain = 10 ** (response_db(taps, passband) / 20)
    assert np.abs(gain - 1).max() <= 2 * delta
    # Each edge keeps its own width: a quarter of the way into the wider transition, one window sized for the
    # narrower edge would already be in its stopband.
    width = max(transition_low_hz, transition_high_hz)
    inside = freq_min - width / 4 if transition_low_hz >= transition_high_hz else freq_max + width / 4
    assert response_db(taps, [inside])[0] > -30


@pytest.mark.parametrize("freq_min, transition_low_hz, stopband_db", [(300.0, 175.0, 45.0), (300.0, 400.0, 60.0)])
def test_design_fir_highpass_meets_spec(freq_min, transition_low_hz, stopband_db):
    taps = design_fir_highpass(freq_min, FS, transition_low_hz, stopband_db)
    assert taps.size % 2 == 1
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-15)
    np.testing.assert_allclose(response_db(taps, [freq_min]), -6.02, atol=0.1)
    assert response_db(taps, np.linspace(0.0, freq_min - transition_low_hz / 2, 2000)).max() <= -stopband_db
    passband = np.linspace(freq_min + transition_low_hz / 2, FS / 2, 4000)
    gain = 10 ** (response_db(taps, passband) / 20)
    assert np.abs(gain - 1).max() <= 2 * 10 ** (-stopband_db / 20)


def test_design_fir_highpass_invalid_arguments():
    with pytest.raises(ValueError, match="transition band"):
        design_fir_highpass(80.0, FS)
    with pytest.raises(ValueError, match="positive"):
        design_fir_highpass(300.0, FS, stopband_db=0.0)


def test_fir_filter_bandpass_attenuates_out_of_band():
    t = np.arange(60_000) / FS
    tones = {50.0: 1.0, 1000.0: 1.0, 12_000.0: 1.0}
    traces = sum(np.sin(2 * np.pi * f * t) for f in tones)[:, np.newaxis].astype(np.float32)
    out = fir_filter(NumpyRecording(traces, FS), bandpass_taps()).get_traces()[10_000:-10_000, 0]
    t = t[10_000:-10_000]
    np.testing.assert_allclose(out, np.sin(2 * np.pi * 1000.0 * t), atol=4 * 10 ** (-45 / 20))


def test_design_fir_bandpass_invalid_arguments():
    with pytest.raises(ValueError, match="transition bands"):
        design_fir_bandpass(80.0, 6000.0, FS)
    with pytest.raises(ValueError, match="transition bands"):
        design_fir_bandpass(300.0, 14_500.0, FS)
    with pytest.raises(ValueError, match="transition bands"):
        design_fir_bandpass(6000.0, 300.0, FS)
    with pytest.raises(ValueError, match="positive"):
        design_fir_bandpass(300.0, 6000.0, FS, stopband_db=0.0)
