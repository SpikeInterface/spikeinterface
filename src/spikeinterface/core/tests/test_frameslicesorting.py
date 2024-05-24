import warnings

import numpy as np
from numpy.testing import assert_raises

from spikeinterface.core import NumpyRecording, NumpySorting


def test_FrameSliceSorting():
    # Single segment sorting, with and without attached recording
    # Since the default end_frame can be set either from the last spike
    # or from the registered recording
    sf = 10
    nsamp = 1000
    max_spike_time = 900
    min_spike_time = 100
    unit_0_train = np.arange(min_spike_time + 10, max_spike_time - 10)
    spike_times = {
        "0": unit_0_train,
        "1": np.arange(min_spike_time, max_spike_time),
    }
    # Sorting with attached rec
    sorting = NumpySorting.from_unit_dict([spike_times], sf)
    rec = NumpyRecording([np.zeros((nsamp, 5))], sampling_frequency=sf)
    sorting.register_recording(rec)
    # Sorting without attached rec
    sorting_norec = NumpySorting.from_unit_dict([spike_times], sf)
    # Sorting with attached rec and exceeding spikes
    sorting_exceeding = NumpySorting.from_unit_dict([spike_times], sf)
    rec_exceeding = NumpyRecording([np.zeros((max_spike_time - 1, 5))], sampling_frequency=sf)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sorting_exceeding.register_recording(rec_exceeding)

    mid_frame = nsamp // 2

    # duration of all slices is mid_frame. Spike trains are re-referenced to the start_time
    # Vary start_frame/end_frame combination
    start_frame, end_frame = None, None
    sub_sorting = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(sub_sorting.get_unit_spike_train("0"), unit_0_train)
    assert sub_sorting._recording.get_total_samples() == nsamp
    sub_sorting_norec = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(sub_sorting_norec.get_unit_spike_train("0"), unit_0_train)
    assert sub_sorting.get_parent() == sorting

    start_frame, end_frame = None, mid_frame
    sub_sorting = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(sub_sorting.get_unit_spike_train("0"), [t for t in unit_0_train if t < mid_frame])
    assert sub_sorting._recording.get_total_samples() == mid_frame
    sub_sorting_norec = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(sub_sorting_norec.get_unit_spike_train("0"), sub_sorting.get_unit_spike_train("0"))

    start_frame, end_frame = mid_frame, None
    sub_sorting = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(
        sub_sorting.get_unit_spike_train("0"), [t - mid_frame for t in unit_0_train if t >= mid_frame]
    )
    assert sub_sorting._recording.get_total_samples() == nsamp - mid_frame
    sub_sorting_norec = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(sub_sorting_norec.get_unit_spike_train("0"), sub_sorting.get_unit_spike_train("0"))

    start_frame, end_frame = mid_frame - 10, mid_frame + 10
    sub_sorting = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(
        sub_sorting.get_unit_spike_train("0"), [t - start_frame for t in unit_0_train if start_frame <= t < end_frame]
    )
    assert sub_sorting._recording.get_total_samples() == 20
    sub_sorting_norec = sorting.frame_slice(start_frame, end_frame)
    assert np.array_equal(sub_sorting_norec.get_unit_spike_train("0"), sub_sorting.get_unit_spike_train("0"))

    # Edge cases: start_frame > end_frame
    assert_raises(Exception, sorting.frame_slice, 100, 90)

    # Edge case: start_frame > max_spike_time
    # Fails without rec (since end_frame is last spike)
    assert_raises(Exception, sorting_norec.frame_slice, max_spike_time + 1, None)
    # Empty sorting with rec
    sub_sorting = sorting.frame_slice(max_spike_time + 1, None)
    assert np.array_equal(sub_sorting.get_unit_spike_train("1"), [])

    # Edge case: end_frame <= min_spike_time
    # Empty sorting
    sub_sorting = sorting.frame_slice(None, min_spike_time)
    assert np.array_equal(sub_sorting.get_unit_spike_train("1"), [])

    # Edge case: start_frame = end_frame
    assert_raises(Exception, sorting.frame_slice, max_spike_time, max_spike_time)

    # Sorting with exceeding spikes
    assert_raises(Exception, sorting_exceeding.frame_slice, None, None)


if __name__ == "__main__":
    test_FrameSliceSorting()
