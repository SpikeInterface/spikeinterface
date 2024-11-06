import pytest
from pathlib import Path

import itertools
from spikeinterface import NumpyRecording
from spikeinterface.core import generate_recording
from spikeinterface.preprocessing.decimate import DecimateRecording
import numpy as np


@pytest.mark.parametrize("num_segments", [1, 2])
@pytest.mark.parametrize("decimation_offset", [0, 1, 5, 21, 101])
@pytest.mark.parametrize("decimation_factor", [1, 7, 50])
def test_decimate(num_segments, decimation_offset, decimation_factor):
    segment_num_samps = [20000, 40000]
    rec = NumpyRecording([np.arange(2 * N).reshape(N, 2) for N in segment_num_samps], 1)

    parent_traces = [rec.get_traces(i) for i in range(num_segments)]

    if decimation_offset >= min(segment_num_samps) or decimation_offset >= decimation_factor:
        with pytest.raises(ValueError):
            decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
        return

    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
    decimated_parent_traces = [parent_traces[i][decimation_offset::decimation_factor] for i in range(num_segments)]

    for start_frame in [0, 1, 5, None, 1000]:
        for end_frame in [0, 1, 5, None, 1000]:
            if start_frame is None:
                start_frame = max(decimated_rec.get_num_samples(i) for i in range(num_segments))
            if end_frame is None:
                end_frame = max(decimated_rec.get_num_samples(i) for i in range(num_segments))

            for i in range(num_segments):
                assert decimated_rec.get_num_samples(i) == decimated_parent_traces[i].shape[0]
                assert np.all(
                    decimated_rec.get_traces(i, start_frame, end_frame)
                    == decimated_parent_traces[i][start_frame:end_frame]
                )

    for i in range(num_segments):
        assert decimated_rec.get_num_samples(i) == decimated_parent_traces[i].shape[0]
        assert np.all(
            decimated_rec.get_traces(i, start_frame, end_frame) == decimated_parent_traces[i][start_frame:end_frame]
        )


def test_decimate_with_times():
    rec = generate_recording(durations=[5, 10])

    # test with times
    times = [rec.get_times(0) + 10, rec.get_times(1) + 20]
    for i, t in enumerate(times):
        rec.set_times(t, i)

    decimation_factor = 2
    decimation_offset = 1
    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)

    for segment_index in range(rec.get_num_segments()):
        assert np.allclose(
            decimated_rec.get_times(segment_index),
            rec.get_times(segment_index)[decimation_offset::decimation_factor],
        )

    # test with t_start
    rec = generate_recording(durations=[5, 10])
    t_starts = [10, 20]
    for t_start, rec_segment in zip(t_starts, rec._recording_segments):
        rec_segment.t_start = t_start
    decimated_rec = DecimateRecording(rec, decimation_factor, decimation_offset=decimation_offset)
    for segment_index in range(rec.get_num_segments()):
        assert np.allclose(
            decimated_rec.get_times(segment_index),
            rec.get_times(segment_index)[decimation_offset::decimation_factor],
        )


if __name__ == "__main__":
    test_decimate()
