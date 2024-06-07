import pytest

from spikeinterface.core import NumpyRecording

from spikeinterface.preprocessing import DepthOrderRecording, depth_order

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
    rec = NumpyRecording(orig_traces, 10)
    rec.set_dummy_probe_from_locations(geom)
    rec_sorted = depth_order(rec)
    assert np.array_equal(rec_sorted.get_channel_ids(), rec.get_channel_ids())
    assert np.array_equal(rec_sorted.get_traces(), orig_traces)

    # geometry out of order with a duplicate
    geom = np.zeros((5, 2), dtype="float32")
    geom[:, 1] = [3, 2, 1, 0, 0]
    rec = NumpyRecording(orig_traces, 10)
    rec.set_dummy_probe_from_locations(geom)
    rec_sorted = depth_order(rec)
    assert np.array_equal(rec_sorted.get_channel_ids(), [3, 4, 2, 1, 0])
    assert np.array_equal(rec_sorted.get_traces(), orig_traces[:, [3, 4, 2, 1, 0]])


if __name__ == "__main__":
    test_depth_order()
