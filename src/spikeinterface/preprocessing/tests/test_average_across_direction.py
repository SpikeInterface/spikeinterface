from spikeinterface.core import NumpyRecording

from spikeinterface.preprocessing import average_across_direction

import numpy as np


def test_average_across_direction():
    # gradient recording with 100 samples and 10 channels
    rec_arr = np.arange(6, dtype="float32")[None, :] * np.ones((100, 1))

    # construct a 2 col geometry
    geom = np.array([[0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3]])

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
    assert np.all(geom_avgy[:, 0] == 0.5)
    assert geom_avgy[0, 1] == 1.0
    assert geom_avgy[1, 1] == 2.0
    assert geom_avgy[2, 1] == 3.0

    # test with channel ids
    # use chans at y in (1, 2)
    traces = rec_avgy.get_traces(channel_ids=["0-1", "2-3"])
    assert traces.shape == (100, 2)
    assert np.all(traces[:, 0] == 0.5)
    assert np.all(traces[:, 1] == 2.5)

    # test averaging across x
    rec_avgx = average_across_direction(rec, direction="x")
    traces = rec_avgx.get_traces()
    assert traces.shape == (100, 2)
    # correct averages
    # the chans at x=0 are [0, 1, 2, 3]
    assert rec_avgx.get_channel_ids()[0] == "0-2-4"
    assert rec_avgx.get_channel_ids()[1] == "1-3-5"

    assert np.all(traces[:, 0] == 2)
    assert np.all(traces[:, 1] == 3)

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
