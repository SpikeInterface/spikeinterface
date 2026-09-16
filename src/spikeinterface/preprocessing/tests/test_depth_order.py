from spikeinterface.core import NumpyRecording
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.testing import check_recordings_equal

from spikeinterface.preprocessing import depth_order
from spikeinterface.preprocessing.depth_order import DepthOrderRecording

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
    rec_sorted = depth_order(recording=rec)
    print(rec_sorted.get_channel_locations())
    assert np.array_equal(rec_sorted.get_channel_ids(), [4, 3, 2, 0, 1])
    assert np.array_equal(rec_sorted.get_traces(), orig_traces[:, [4, 3, 2, 0, 1]])


def test_depth_order_parent_recording_from_dict():
    """
    `DepthOrderRecording` inherits `ChannelSliceRecording.__init__`, which always serializes the
    wrapped recording under the legacy `parent_recording` key in `_kwargs`. `BaseExtractor.from_dict`
    must keep reconstructing it from that key even though `__init__`'s first parameter is now
    named `recording`.
    """
    orig_traces = np.arange(5, dtype="float32")[None, :] * np.ones((100, 1))
    geom = np.zeros((5, 2), dtype="float32")
    geom[:, 1] = [2, 3, 1, 0, -1]
    recording = NumpyRecording(orig_traces, 10)
    recording.set_dummy_probe_from_locations(geom)

    recording_sorted = DepthOrderRecording(recording=recording)
    # We artificially rename 'recording' to 'parent_recording' in the kwargs to simulate legacy behavior
    recording_sorted._kwargs["parent_recording"] = recording_sorted._kwargs.pop("recording")
    reloaded = BaseExtractor.from_dict(recording_sorted.to_dict())
    check_recordings_equal(recording_sorted, reloaded)


if __name__ == "__main__":
    test_depth_order()
