from spikeinterface.core import NumpyRecording

from spikeinterface.preprocessing import depth_order

import numpy as np


def test_depth_order():
    # gradient recording with 100 samples and 10 channels
    orig_traces = np.arange(5, dtype="float32")[None, :] * np.ones((100, 1))

    # geometry in order
    geom = np.zeros((5, 2), dtype="float32")
    geom[:, 1] = np.arange(5)
    rec = NumpyRecording(orig_traces, 10)
    rec.set_dummy_probe_from_locations(geom)
    rec_sorted = depth_order(rec)
    assert np.array_equal(rec_sorted.get_channel_ids(), rec.get_channel_ids())
    assert np.array_equal(rec_sorted.get_traces(), orig_traces)

    # geometry flat -- needs to be stable!
    geom = np.zeros((5, 2), dtype="float32")
    geom[:, 0] = np.arange(5)
    rec = NumpyRecording(orig_traces, 10)
    rec.set_dummy_probe_from_locations(geom)
    rec_sorted = depth_order(rec)
    assert np.array_equal(rec_sorted.get_channel_ids(), rec.get_channel_ids())
    assert np.array_equal(rec_sorted.get_traces(), orig_traces)

    # geometry out of order
    geom = np.zeros((5, 2), dtype="float32")
    geom[:, 1] = [2, 3, 1, 0, -1]
    rec = NumpyRecording(orig_traces, 10)
    rec.set_dummy_probe_from_locations(geom)
    print(rec.get_channel_locations())
    rec_sorted = depth_order(rec)
    print(rec_sorted.get_channel_locations())
    assert np.array_equal(rec_sorted.get_channel_ids(), [4, 3, 2, 0, 1])
    assert np.array_equal(rec_sorted.get_traces(), orig_traces[:, [4, 3, 2, 0, 1]])


if __name__ == "__main__":
    test_depth_order()
