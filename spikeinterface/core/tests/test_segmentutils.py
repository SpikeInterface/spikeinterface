import pytest
import numpy as np

from spikeinterface.core import (
    append_recordings, AppendSegmentRecording,
    concatenate_recordings, ConcatenateSegmentRecording,
    append_sortings, AppendSegmentSorting)

from spikeinterface.core import NumpyRecording, NumpySorting


def test_append_concatenate_recordings():
    traces = np.zeros((1000, 5), dtype='float64')
    traces[:] = np.arange(1000)[:, None]
    sampling_frequency = 30000
    rec0 = NumpyRecording([traces] * 3, sampling_frequency)
    rec1 = NumpyRecording([traces] * 2, sampling_frequency)

    # append
    rec = append_recordings([rec0, rec1])
    #  print(rec)
    assert rec.get_num_segments() == 5
    for segment_index in range(5):
        traces = rec.get_traces(segment_index=segment_index)
        assert rec.get_num_samples(segment_index) == 1000

    # concatenate
    rec = concatenate_recordings([rec0, rec1])
    #  print(rec)
    assert rec.get_num_samples(0) == 5 * 1000
    assert rec.get_num_segments() == 1
    assert rec.get_times(0).size == 5000

    # case one segment
    traces = rec.get_traces(start_frame=0, end_frame=15)
    assert np.array_equal(traces[:, 0], np.arange(0, 15))
    traces = rec.get_traces(start_frame=500, end_frame=750)
    assert np.array_equal(traces[:, 0], np.arange(500, 750))

    # case on limit
    traces = rec.get_traces(start_frame=1000, end_frame=2000)
    assert traces.shape == (1000, 5)
    assert np.array_equal(traces[:, 0], np.arange(0, 1000, dtype='float64'))

    # case total
    traces = rec.get_traces(start_frame=None, end_frame=None)
    assert traces.shape == (5000, 5)
    assert traces[1000, 0] == 0

    # several segments
    traces = rec.get_traces(start_frame=50, end_frame=4500)
    assert traces.shape == (4450, 5)
    assert traces[0, 0] == 50
    assert traces[-1, 0] == 499


def test_append_sortings():
    sampling_frequency = 30000.
    times = np.arange(0, 1000, 10)
    labels = np.zeros(times.size, dtype='int64')
    labels[0::3] = 0
    labels[1::3] = 1
    labels[2::3] = 2
    sorting0 = NumpySorting.from_times_labels(
        [times] * 3, [labels] * 3, sampling_frequency)
    sorting1 = NumpySorting.from_times_labels(
        [times] * 2, [labels] * 2, sampling_frequency)

    sorting = append_sortings([sorting0, sorting1])
    # print(sorting)
    assert sorting.get_num_segments() == 5


if __name__ == '__main__':
    test_append_concatenate_recordings()
    test_append_sortings()
