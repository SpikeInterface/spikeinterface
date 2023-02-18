from spikeinterface.core import generate_sorting


def test_FrameSliceSorting():
    fs = 30000
    duration = 10
    sort = generate_sorting(num_units=10, durations=[duration], sampling_frequency=fs)

    mid_frame = (duration * fs) // 2
    # duration of all slices is mid_frame. Spike trains are re-referenced to the start_time
    sub_sort = sort.frame_slice(None, None)
    for u in sort.get_unit_ids():
        assert len(sort.get_unit_spike_train(u)) == len(sub_sort.get_unit_spike_train(u))

    sub_sort = sort.frame_slice(None, mid_frame)
    for u in sort.get_unit_ids():
        assert max(sub_sort.get_unit_spike_train(u)) <= mid_frame

    sub_sort = sort.frame_slice(mid_frame, None)
    for u in sort.get_unit_ids():
        assert max(sub_sort.get_unit_spike_train(u)) <= mid_frame

    sub_sort = sort.frame_slice(mid_frame - mid_frame // 2, mid_frame + mid_frame // 2)
    for u in sort.get_unit_ids():
        assert max(sub_sort.get_unit_spike_train(u)) <= mid_frame


if __name__ == "__main__":
    test_FrameSliceSorting()
