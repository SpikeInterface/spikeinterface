import pytest
import numpy as np

from spikeinterface.core import FrameSliceRecording, NumpyRecording


def test_FrameSliceRecording():
    traces = np.zeros((1000, 5), dtype='float64')
    traces[:] = np.arange(1000)[:, None]
    sampling_frequency = 30000
    rec = NumpyRecording([traces], sampling_frequency)

    sub_rec = rec.frame_slice(None, None)
    assert sub_rec.get_num_samples(0) == 1000
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(0, 1000, dtype='float64'))

    sub_rec = rec.frame_slice(None, 10)
    assert sub_rec.get_num_samples(0) == 10
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(0, 10, dtype='float64'))

    sub_rec = rec.frame_slice(900, 1000)
    assert sub_rec.get_num_samples(0) == 100
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(900, 1000, dtype='float64'))

    sub_rec = rec.frame_slice(10, 85)
    assert sub_rec.get_num_samples(0) == 75
    traces = sub_rec.get_traces()
    assert np.array_equal(traces[:, 0], np.arange(10, 85, dtype='float64'))


if __name__ == '__main__':
    test_FrameSliceRecording()
