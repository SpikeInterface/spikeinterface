import warnings

import numpy as np
import pytest
from numpy.testing import assert_raises

from spikeinterface.core import (
    NumpyRecording,
    NumpySorting,
    append_recordings,
    append_sortings,
    concatenate_recordings,
    concatenate_sortings,
    select_segment_recording,
    split_recording,
    split_sorting,
)
from spikeinterface.core.testing import check_sorted_arrays_equal


def test_append_concatenate_recordings():
    base_traces = np.zeros((1000, 5), dtype="float64")
    base_traces[:] = np.arange(1000)[:, None]
    sampling_frequency = 30000
    rec0 = NumpyRecording([base_traces] * 3, sampling_frequency)
    rec1 = NumpyRecording([base_traces] * 2, sampling_frequency)

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
    assert np.array_equal(traces, base_traces[:15])
    traces = rec.get_traces(start_frame=0, end_frame=1000)
    assert np.array_equal(traces, base_traces)
    traces = rec.get_traces(start_frame=500, end_frame=750)
    assert np.array_equal(traces, base_traces[500:750])

    # case on limit
    traces = rec.get_traces(start_frame=1000, end_frame=2000)
    assert traces.shape == (1000, 5)
    assert np.array_equal(traces, base_traces)

    # case total
    traces = rec.get_traces(start_frame=None, end_frame=None)
    assert traces.shape == (5000, 5)
    assert np.array_equal(traces[:1000], base_traces)
    assert np.array_equal(traces[1000:2000], base_traces)
    assert np.array_equal(traces[4000:5000], base_traces)

    # several segments
    traces = rec.get_traces(start_frame=50, end_frame=4500)
    assert traces.shape == (4450, 5)
    assert np.array_equal(traces[0:10], base_traces[50:60])
    assert np.array_equal(traces[-10:], base_traces[490:500])
    # across segments
    assert np.array_equal(traces[949, :], base_traces[-1, :])
    assert np.array_equal(traces[950, :], base_traces[0, :])


def test_split_recordings():
    traces = np.zeros((1000, 5), dtype="float64")
    traces[:] = np.arange(1000)[:, None]
    sampling_frequency = 30000
    rec0 = NumpyRecording([traces] * 3, sampling_frequency)

    rec_list = split_recording(rec0)
    rec1 = select_segment_recording(rec0, segment_indices=[1])
    rec2 = rec0.select_segments(segment_indices=[1, 2])
    rec3 = select_segment_recording(rec0, segment_indices=2)

    for i, rec in enumerate(rec_list):
        assert rec.get_num_segments() == 1
        assert np.allclose(rec.get_traces(), rec0.get_traces(segment_index=i))

    assert rec1.get_num_segments() == 1
    assert np.allclose(rec1.get_traces(), rec0.get_traces(segment_index=1))

    assert np.allclose(rec2.get_traces(segment_index=0), rec0.get_traces(segment_index=1))
    assert np.allclose(rec2.get_traces(segment_index=1), rec0.get_traces(segment_index=2))
    assert np.allclose(rec3.get_traces(), rec0.get_traces(segment_index=2))


def test_append_concatenate_sortings():
    sampling_frequency = 30000.0
    nsamp0 = 1000
    nsamp1 = 1500
    times0 = np.arange(0, nsamp0)
    times1 = np.arange(0, nsamp1 - 1) + 1
    labels0 = np.zeros(times0.size, dtype="int64")
    labels0[0::3] = 0
    labels0[1::3] = 1
    labels0[2::3] = 2
    labels1 = np.zeros(times1.size, dtype="int64")
    labels1[0::3] = 0
    labels1[1::3] = 1
    labels1[2::3] = 2

    # Multisegment sortings
    sorting0 = NumpySorting.from_times_labels([times0] * 3, [labels0] * 3, sampling_frequency)
    sorting1 = NumpySorting.from_times_labels([times1] * 2, [labels1] * 2, sampling_frequency)
    sorting_list = [sorting0, sorting1]
    # Associated multisegment recordings
    traces0 = np.zeros((nsamp0, 5), dtype="float64")
    traces1 = np.zeros((nsamp1, 5), dtype="float64")
    rec0 = NumpyRecording([traces0] * 3, sampling_frequency)
    rec1 = NumpyRecording([traces1] * 2, sampling_frequency)
    # Monosegemnt sortings
    sorting0_mono = NumpySorting.from_times_labels([times0], [labels0], sampling_frequency)
    sorting1_mono = NumpySorting.from_times_labels([times1], [labels1], sampling_frequency)
    sorting_list_mono = [sorting0_mono, sorting1_mono]
    # Associated multisegment recordings
    traces0 = np.zeros((nsamp0, 5), dtype="float64")
    traces1 = np.zeros((nsamp1, 5), dtype="float64")
    rec0_mono = NumpyRecording([traces0], sampling_frequency)
    rec1_mono = NumpyRecording([traces1], sampling_frequency)
    # Recording too short
    traces0_short = np.zeros((nsamp0 - 1, 5), dtype="float64")
    rec0_mono_short = NumpyRecording([traces0_short], sampling_frequency)

    # Append
    # Append multisegment
    sorting = append_sortings(sorting_list)
    # print(sorting)
    assert sorting.get_num_segments() == 5
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=0),
        times0[0::3],
    )
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=3),
        times1[0::3],
    )
    # Append monosegment
    sorting_mono = append_sortings(sorting_list_mono)
    assert sorting_mono.get_num_segments() == 2
    assert np.array_equal(
        sorting_mono.get_unit_spike_train(unit_id=0, segment_index=0),
        times0[0::3],
    )
    assert np.array_equal(
        sorting_mono.get_unit_spike_train(unit_id=0, segment_index=1),
        times1[0::3],
    )

    # Concat
    # Fails without registered recording
    assert_raises(Exception, concatenate_sortings, sorting_list)
    assert_raises(Exception, concatenate_sortings, sorting_list_mono)
    # Fails with total_samples_list for multisegment
    assert_raises(Exception, concatenate_sortings, sorting_list, total_samples_list="dummy")
    # Succeeds without registered recording for mono
    sorting_mono_norec = concatenate_sortings(sorting_list_mono, total_samples_list=[nsamp0, nsamp1])
    assert sorting_mono_norec.get_num_segments() == 1
    # Fails when excess spikes with total_samples_list
    assert_raises(Exception, concatenate_sortings, sorting_list, total_samples_list=[nsamp0 - 1, nsamp1])
    # Fails when excess spikes with registered recording
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sorting0_mono.register_recording(rec0_mono_short)
    assert_raises(Exception, concatenate_sortings, [sorting0_mono])

    # Fails

    # With registered recording
    sorting0.register_recording(rec0)
    sorting1.register_recording(rec1)
    sorting0_mono.register_recording(rec0_mono)
    sorting1_mono.register_recording(rec1_mono)
    sorting = concatenate_sortings(sorting_list)
    sorting_mono = concatenate_sortings(sorting_list_mono)
    # Check Segments
    assert sorting.get_num_segments() == 1
    assert sorting_mono.get_num_segments() == 1
    # Check nsamps
    assert sorting.get_total_samples() == 3 * nsamp0 + 2 * nsamp1
    assert sorting_mono.get_total_samples() == nsamp0 + nsamp1
    assert sorting.get_num_samples() == 3 * nsamp0 + 2 * nsamp1
    assert sorting_mono.get_num_samples() == nsamp0 + nsamp1

    # Check spike trains
    unit_0_train = np.concatenate(
        (
            times0[::3],
            nsamp0 + times0[::3],
            2 * nsamp0 + times0[::3],
            3 * nsamp0 + times1[::3],
            (3 * nsamp0 + nsamp1) + times1[::3],
        ),
        axis=0,
    )

    # Within one segment
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=None, start_frame=0, end_frame=1000),
        [t for t in unit_0_train if 0 <= t < 1000],
    )  # Full first
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=None, start_frame=0, end_frame=300),
        [t for t in unit_0_train if 0 <= t < 300],
    )  # Part first
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=None, start_frame=500, end_frame=750),
        [t for t in unit_0_train if 500 <= t < 750],
    )  # Part first
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=None, start_frame=1000, end_frame=2000),
        [t for t in unit_0_train if 1000 <= t < 2000],
    )  # Full 2nd segment
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=None, start_frame=4500, end_frame=6000),
        [t for t in unit_0_train if 4500 <= t < 6000],
    )  # Full last segment
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=None, start_frame=4550, end_frame=5000),
        [t for t in unit_0_train if 4550 <= t < 5000],
    )  # Part of last segmetn

    # Total
    assert np.array_equal(sorting.get_unit_spike_train(unit_id=0), unit_0_train)

    # Across segments
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, segment_index=None, start_frame=950, end_frame=1050),
        [t for t in unit_0_train if 950 <= t < 1050],
    )

    # Check same result with and without rec for mono segment
    assert np.array_equal(
        sorting_mono.get_unit_spike_train(unit_id=0),
        sorting_mono_norec.get_unit_spike_train(unit_id=0),
    )

    # Slicing and concatenating back
    # Full sorting
    sorting_beginning = sorting.frame_slice(None, 1000)
    sorting_end = sorting.frame_slice(1000, None)
    sorting_concat = concatenate_sortings([sorting_beginning, sorting_end])
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0),
        sorting_concat.get_unit_spike_train(unit_id=0),
    )
    # Slice in the middle
    sorting_beginning = sorting.frame_slice(50, 1000)
    sorting_end = sorting.frame_slice(1000, 2000)
    sorting_concat = concatenate_sortings([sorting_beginning, sorting_end])
    assert np.array_equal(
        sorting.get_unit_spike_train(unit_id=0, start_frame=50, end_frame=2000) - 50,
        sorting_concat.get_unit_spike_train(unit_id=0),
    )

    # Time conversion
    # Basic (no time vector)
    assert np.array_equal(sorting.get_unit_spike_train(unit_id=0, return_times=True), unit_0_train / sampling_frequency)

    # Time conversion with non-trivial t_start/time_vector
    sorting0 = NumpySorting.from_times_labels([[0, 1]], [[0, 0]], sampling_frequency)
    sorting1 = NumpySorting.from_times_labels([[0, 1]], [[0, 0]], sampling_frequency)
    traces = np.zeros((2, 5), dtype="float64")
    rec0 = NumpyRecording([traces], sampling_frequency)
    rec1 = NumpyRecording([traces], sampling_frequency)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec0.set_times(np.array([1000, 1010]))
        rec1.set_times(np.array([2000, 2010]))
    sorting0.register_recording(rec0)
    sorting1.register_recording(rec1)
    # Actually, non-trivial time is not handled by concatenate_recordings
    assert_raises(Exception, concatenate_sortings, [sorting0, sorting1], ignore_times=False)
    # sorting = concatenate_sortings([sorting0, sorting1], ignore_times=False)
    # assert np.array_equal(
    #     sorting.get_unit_spike_train(unit_id=0, return_times=True),
    #     [1000, 1010, 2000, 2010],
    # )


if __name__ == "__main__":
    test_append_concatenate_recordings()
    test_append_concatenate_sortings()
