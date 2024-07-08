import pytest
from pathlib import Path

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import NumpyRecording

from spikeinterface.preprocessing import AverageAcrossDirectionRecording, average_across_direction

import numpy as np


def test_average_across_direction():
    # gradient recording with 100 samples and 10 channels
    rec_arr = np.arange(6, dtype="float32")[None, :] * np.ones((100, 1))

    # test geometry with chans at the same y position
    geom = np.zeros((6, 2), dtype="float32")
    geom[[0, 1], 1] = 1
    geom[[2, 3], 1] = 2
    geom[[4, 5], 1] = 3

    # have the x position change as well to test the geometry
    geom[[4, 5], 0] = [1, 2]

    rec = NumpyRecording(rec_arr, 10)
    rec.set_dummy_probe_from_locations(geom)

    # test averaging across y
    rec_avgy = average_across_direction(rec)
    traces = rec_avgy.get_traces()
    assert traces.shape == (100, 3)
    # correct averages
    assert np.all(traces[:, 0] == 0.5)
    assert np.all(traces[:, 1] == 2.5)
    assert np.all(traces[:, 2] == 4.5)
    geom_avgy = rec_avgy.get_channel_locations()
    assert np.all(geom_avgy[:2, 0] == 0)
    assert np.all(geom_avgy[2, 0] == 1.5)

    # test averaging across x
    rec_avgx = average_across_direction(rec, direction="x")
    traces = rec_avgx.get_traces()
    assert traces.shape == (100, 3)
    # correct averages
    # the chans at x=0 are [0, 1, 2, 3]
    assert rec_avgx.get_channel_ids()[0] == "0-1-2-3"
    assert np.all(traces[:, 0] == 1.5)
    assert np.all(traces[:, 1] == 4)
    assert np.all(traces[:, 2] == 5)

    # int16 test
    rec = NumpyRecording(rec_arr.astype("int16"), 10)
    rec.set_dummy_probe_from_locations(geom)
    rec_avgy = average_across_direction(rec, dtype=None)
    try:
        traces = rec_avgy.get_traces()
        assert False
    except np.core._exceptions._UFuncOutputCastingError:
        pass


if __name__ == "__main__":
    test_average_across_direction()
