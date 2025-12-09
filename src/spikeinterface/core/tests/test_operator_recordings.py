import pytest
import numpy as np

import spikeinterface as si


@pytest.fixture
def recording():
    recording = si.generate_recording(durations=[10, 20], num_channels=4, sampling_frequency=20000)
    return recording


def test_sum_recordings(recording):
    rec_sum = recording + recording
    for seg_index in range(rec_sum.get_num_segments()):
        traces_orig = recording.get_traces(segment_index=seg_index)
        traces_sum = rec_sum.get_traces(segment_index=seg_index)
        np.testing.assert_array_equal(traces_sum, traces_orig * 2)


def test_subtract_recordings(recording):
    rec_sub = recording - recording
    for seg_index in range(rec_sub.get_num_segments()):
        traces_sub = rec_sub.get_traces(segment_index=seg_index)
        np.testing.assert_array_equal(traces_sub, np.zeros_like(traces_sub))


def test_operator_combo(recording):
    rec_combo = recording - recording + recording - recording + recording
    for seg_index in range(rec_combo.get_num_segments()):
        traces_orig = recording.get_traces(segment_index=seg_index)
        traces_combo = rec_combo.get_traces(segment_index=seg_index)
        np.testing.assert_array_equal(traces_combo, traces_orig)


def test_errors(recording):
    recording2 = si.generate_recording(durations=[10, 20], num_channels=4, sampling_frequency=10000)
    with pytest.raises(AssertionError):
        _ = recording + recording2
    with pytest.raises(AssertionError):
        _ = recording - recording2

    recording_times = recording.clone()
    for segment_index in range(recording_times.get_num_segments()):
        recording_times.set_times(
            recording_times.get_times(segment_index=segment_index) + (segment_index + 1) * 5,
            segment_index=segment_index,
        )
    with pytest.raises(AssertionError):
        _ = recording + recording_times
