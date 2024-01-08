import pytest
from pathlib import Path

import itertools
from spikeinterface import NumpyRecording
from spikeinterface.core import generate_recording
from spikeinterface.preprocessing.decimate import DecimateRecording
import numpy as np


@pytest.mark.parametrize("decimation_frame_start", [0, 1, 9, 10, 11, 200])
@pytest.mark.parametrize("decimation_frame_step", [1, 9, 10, 11, 200])
@pytest.mark.parametrize("start_frame", [0, 1, 5, 20])
@pytest.mark.parametrize("end_frame", [None, 1, 5, 20])
def test_decimate(decimation_frame_start, decimation_frame_step, start_frame, end_frame):
    rec = generate_recording()

    N = 101
    rec = NumpyRecording([np.arange(N).reshape(N,1)], 1)
    parent_traces = rec.get_traces()

    decimated_rec = DecimateRecording(rec, decimation_frame_step, decimation_frame_start=decimation_frame_start)
    decimated_parent_traces = parent_traces[decimation_frame_start::decimation_frame_step]

    if start_frame is None:
        start_frame = len(decimated_parent_traces)

    assert np.all(
        decimated_rec.get_traces(0, start_frame, end_frame) == decimated_parent_traces[start_frame:end_frame]
    )


if __name__ == "__main__":
    test_decimate()
