import pytest
from pathlib import Path

import itertools
from spikeinterface import NumpyRecording
from spikeinterface.core import generate_recording
from spikeinterface.preprocessing.decimate import DecimateRecording
import numpy as np


@pytest.mark.parametrize("N_segments", [1, 2])
@pytest.mark.parametrize("decimation_offset", [0, 1, 9, 10, 11, 100, 101])
@pytest.mark.parametrize("decimation_factor", [1, 9, 10, 11, 100, 101])
@pytest.mark.parametrize("start_frame", [0, 1, 5, None, 1000])
@pytest.mark.parametrize("end_frame", [0, 1, 5, None, 1000])
def test_decimate(N_segments, decimation_offset, decimation_factor, start_frame, end_frame):
    rec = generate_recording()

    segment_num_samps = [101 + i for i in range(N_segments)]

    rec = NumpyRecording([np.arange(2 * N).reshape(N, 2) for N in segment_num_samps], 1)

    parent_traces = [rec.get_traces(i) for i in range(N_segments)]

    if decimation_offset >= min(segment_num_samps) or decimation_offset >= decimation_factor:
        with pytest.raises(ValueError):
            decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
        return

    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
    decimated_parent_traces = [parent_traces[i][decimation_offset::decimation_factor] for i in range(N_segments)]

    if start_frame is None:
        start_frame = max(decimated_rec.get_num_samples(i) for i in range(N_segments))
    if end_frame is None:
        end_frame = max(decimated_rec.get_num_samples(i) for i in range(N_segments))

    for i in range(N_segments):
        assert decimated_rec.get_num_samples(i) == decimated_parent_traces[i].shape[0]
        assert np.all(
            decimated_rec.get_traces(i, start_frame, end_frame) == decimated_parent_traces[i][start_frame:end_frame]
        )


if __name__ == "__main__":
    test_decimate()
