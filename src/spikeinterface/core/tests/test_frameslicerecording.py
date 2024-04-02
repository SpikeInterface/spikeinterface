import pytest
import numpy as np

from spikeinterface.core import FrameSliceRecording, NumpyRecording


def test_FrameSliceRecording():
    traces = np.zeros((1000, 5), dtype="float64")
    traces[:] = np.arange(1000)[:, None]
    sampling_frequency = 30000
    rec = NumpyRecording([traces], sampling_frequency)
    times0 = rec.get_times(0)

    sub_rec = rec.frame_slice(None, None)
    assert sub_rec.get_num_samples(0) == 1000
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(0, 1000, dtype="float64"))
    sub_times0 = sub_rec.get_times(0)
    assert np.allclose(times0, sub_times0)
    assert sub_rec.get_parent() == rec

    sub_rec = rec.frame_slice(None, 10)
    assert sub_rec.get_num_samples(0) == 10
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(0, 10, dtype="float64"))
    sub_times0 = sub_rec.get_times(0)
    assert np.allclose(times0[:10], sub_times0)

    sub_rec = rec.frame_slice(900, 1000)
    assert sub_rec.get_num_samples(0) == 100
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(900, 1000, dtype="float64"))
    sub_times0 = sub_rec.get_times(0)
    assert np.allclose(times0[900:1000], sub_times0)

    sub_rec = rec.frame_slice(10, 85)
    assert sub_rec.get_num_samples(0) == 75
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(10, 85, dtype="float64"))
    sub_times0 = sub_rec.get_times(0)
    assert np.allclose(times0[10:85], sub_times0)


if __name__ == "__main__":
    test_FrameSliceRecording()
