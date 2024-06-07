from spikeinterface.core import NumpyRecording

from spikeinterface.preprocessing import DirectionalDerivativeRecording, directional_derivative

import numpy as np


def test_directional_derivative():
    # gradient recording with 100 samples and 10 channels
    rec = np.arange(10, dtype="float32")[None, :] * np.ones((100, 1))

    geom_y = np.zeros((10, 2), dtype="float32")
    geom_y[:, 1] = np.arange(10)
    rec_y = NumpyRecording(rec, 10)
    rec_y.set_dummy_probe_from_locations(geom_y)
    rec_y_ddy = directional_derivative(rec_y)
    traces = rec_y_ddy.get_traces()
    assert (traces == 1).all()
    rec_y_ddx = directional_derivative(rec_y, direction="x")
    traces = rec_y_ddx.get_traces()
    assert (traces == 0).all()

    geom_x = np.zeros((10, 2), dtype="float32")
    geom_x[:, 0] = np.arange(10)
    rec_x = NumpyRecording(rec, 10)
    rec_x.set_dummy_probe_from_locations(geom_x)
    rec_x_ddx = directional_derivative(rec_x, direction="x")
    traces = rec_x_ddx.get_traces()
    assert (traces == 1).all()
    rec_x_ddy = directional_derivative(rec_x, direction="y")
    traces = rec_x_ddy.get_traces()
    assert (traces == 0).all()

    # int16 test
    geom_x = np.zeros((10, 2), dtype="float32")
    geom_x[:, 0] = np.arange(10)
    rec_x = NumpyRecording(rec.astype("int16"), 10)
    rec_x.set_dummy_probe_from_locations(geom_x)
    rec_x_ddx = directional_derivative(rec_x, direction="x", dtype=None)
    traces = rec_x_ddx.get_traces()
    assert (traces == 1).all()
    rec_x_ddy = directional_derivative(rec_x, direction="y")
    traces = rec_x_ddy.get_traces()
    assert (traces == 0).all()


if __name__ == "__main__":
    test_directional_derivative()
