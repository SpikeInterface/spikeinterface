import numpy as np
import pytest
from pathlib import Path
from spikeinterface.core import load_extractor, set_global_tmp_folder
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.core.generate import generate_recording
from spikeinterface.preprocessing import gaussian_filter
from numpy.testing import assert_allclose
from spikeinterface.core import NumpyRecording


def test_filter_gaussian(tmp_path):
    recording = generate_recording(num_channels=3)
    recording.annotate(is_filtered=True)
    recording = recording.save(folder=tmp_path / "recording")

    rec_filtered = gaussian_filter(recording)

    assert rec_filtered.dtype == recording.dtype
    assert rec_filtered.get_traces(segment_index=0, end_frame=100).dtype == rec_filtered.dtype
    assert rec_filtered.get_traces(segment_index=0, end_frame=600).shape == (600, 3)
    assert rec_filtered.get_traces(segment_index=0, start_frame=100, end_frame=600).shape == (500, 3)
    assert rec_filtered.get_traces(segment_index=1, start_frame=rec_filtered.get_num_frames(1) - 200).shape == (200, 3)

    # Check dumpability
    saved_loaded = load_extractor(rec_filtered.to_dict())
    check_recordings_equal(rec_filtered, saved_loaded, return_scaled=False)

    saved_1job = rec_filtered.save(folder=tmp_path / "1job")
    saved_2job = rec_filtered.save(folder=tmp_path / "2job", n_jobs=2, chunk_duration="1s")

    for seg_idx in range(rec_filtered.get_num_segments()):
        original_trace = rec_filtered.get_traces(seg_idx)
        saved1_trace = saved_1job.get_traces(seg_idx)
        saved2_trace = saved_2job.get_traces(seg_idx)

        assert np.allclose(original_trace[60:-60], saved1_trace[60:-60], rtol=1e-3, atol=1e-3)
        assert np.allclose(original_trace[60:-60], saved2_trace[60:-60], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("freq_min", [None, 10, 50, 100])
@pytest.mark.parametrize("freq_max", [None, 10, 50, 100])
def test_bandpower(freq_min, freq_max, debug=False):
    if freq_min is None and freq_max is None:
        return
    if freq_min is not None and freq_max is not None:
        if freq_min >= freq_max:
            return

    fs = 1000
    T = 300
    t = np.arange(0, T, 1 / fs)

    rng = np.random.default_rng()
    x = rng.standard_normal(T * fs)

    rec = NumpyRecording([x.reshape(len(x), 1)], fs)
    rec_filt = gaussian_filter(rec, freq_min=freq_min, freq_max=freq_max)

    # Welch power density
    trace = rec.get_traces()[:, 0]
    trace_filt = rec_filt.get_traces(0)[:, 0]
    import scipy

    f, Pxx = scipy.signal.welch(trace, fs=fs)
    _, Pxx_filt = scipy.signal.welch(trace_filt, fs=fs)

    if debug:
        import matplotlib.pyplot as plt

        plt.plot(f, Pxx, label="Welch original")
        plt.plot(f, Pxx_filt, label="Welch gaussian filter")
        plt.plot(f, Pxx - Pxx_filt, label="difference")
        plt.xlabel("Freq (Hz)")
        plt.legend()

    # Absolute and relative diff for assert_allclose
    atol = np.mean(Pxx) / 10
    rtol = 1 / 10

    def assert_closeby(Pxx_filt, Pxx, freq, above_or_below):
        ind = np.argmax(f > freq) - 1
        if above_or_below == "above":
            assert_allclose(Pxx_filt[ind:], Pxx[ind:], rtol=rtol, atol=atol)
        else:
            assert_allclose(Pxx_filt[:ind], Pxx[:ind], rtol=rtol, atol=atol)

    def assert_null(Pxx_filt, freq, above_or_below):
        ind = np.argmax(f > freq) - 1
        if above_or_below == "above":
            assert_allclose(Pxx_filt[ind:], 0, rtol=0, atol=atol)
        else:
            assert_allclose(Pxx_filt[:ind], 0, rtol=0, atol=atol)

    if freq_max is None:
        # high pass
        assert_closeby(Pxx_filt, Pxx, freq_min * 3, "above")
        assert_null(Pxx_filt, freq_min * 0.5, "below")

    elif freq_min is None:
        # Low pass
        assert_closeby(Pxx_filt, Pxx, freq_max * 0.5, "below")
        assert_null(Pxx_filt, freq_max * 2, "above")

    else:
        # Don't test in passband because strong attenuation
        assert_null(Pxx_filt, freq_min * 0.5, "below")
        assert_null(Pxx_filt, freq_max * 2, "above")


if __name__ == "__main__":
    test_filter_gaussian()
    test_bandpower()
