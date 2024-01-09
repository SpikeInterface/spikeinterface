import pytest
from pathlib import Path

import itertools
from spikeinterface import NumpyRecording
from spikeinterface.core import generate_recording
from spikeinterface.preprocessing.decimate import DecimateRecording
import numpy as np


@pytest.mark.parametrize("decimation_offset", [0, 1, 9, 10, 11, 100, 101])
@pytest.mark.parametrize("decimation_factor", [1, 9, 10, 11, 100, 101])
@pytest.mark.parametrize("start_frame", [0, 1, 5, None, 1000])
@pytest.mark.parametrize("end_frame", [0, 1, 5, None, 1000])
def test_decimate(decimation_offset, decimation_factor, start_frame, end_frame):
    rec = generate_recording()

    N = 101
    rec = NumpyRecording([np.arange(2 * N).reshape(N, 2)], 1)
    parent_traces = rec.get_traces()

    if decimation_offset >= N or decimation_offset >= decimation_factor:
        with pytest.raises(ValueError):
            decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
        return

    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
    decimated_parent_traces = parent_traces[decimation_offset::decimation_factor]

    if start_frame is None:
        start_frame = decimated_rec.get_num_samples()
    if end_frame is None:
        end_frame = decimated_rec.get_num_samples()

    assert decimated_rec.get_num_samples() == decimated_parent_traces.shape[0]
    assert np.all(decimated_rec.get_traces(0, start_frame, end_frame) == decimated_parent_traces[start_frame:end_frame])


if __name__ == "__main__":
    test_decimate()
