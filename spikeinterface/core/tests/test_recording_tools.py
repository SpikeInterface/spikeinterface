import numpy as np

from spikeinterface.core import generate_recording

from spikeinterface.core.recording_tools import (get_random_data_chunks, get_chunk_with_margin,
                                                 get_closest_channels, get_channel_distances, get_noise_levels,
                                                 order_channels_by_depth)


def test_get_random_data_chunks():
    rec = generate_recording(num_channels=1, sampling_frequency=1000., durations=[10., 20.])
    chunks = get_random_data_chunks(rec, num_chunks_per_segment=50, chunk_size=500, seed=0)
    assert chunks.shape == (50000, 1)

    

def test_get_closest_channels():
    rec = generate_recording(num_channels=32, sampling_frequency=1000., durations=[0.1])
    closest_channels_inds, distances = get_closest_channels(rec)
    closest_channels_inds, distances = get_closest_channels(rec, num_channels=4)
    
    dist = get_channel_distances(rec)

def test_get_noise_levels():
    rec = generate_recording(num_channels=2, sampling_frequency=1000., durations=[60.])

    noise_levels = get_noise_levels(rec, return_scaled=False)
    print(noise_levels)

    rec.set_channel_gains(0.1)
    rec.set_channel_offsets(0)
    noise_levels = get_noise_levels(rec, return_scaled=True)
    print(noise_levels)

def test_get_chunk_with_margin():
    rec = generate_recording(num_channels=1, sampling_frequency=1000., durations=[10.])
    rec_seg = rec._recording_segments[0]
    length = rec_seg.get_num_samples()

    #  rec_segment, start_frame, end_frame, channel_indices, sample_margin

    traces, l, r = get_chunk_with_margin(rec_seg, None, None, None, 10)
    assert l == 0 and r == 0

    traces, l, r = get_chunk_with_margin(rec_seg, 5, None, None, 10)
    assert l == 5 and r == 0

    traces, l, r = get_chunk_with_margin(rec_seg, length - 1000, length - 5, None, 10)
    assert l == 10 and r == 5
    assert traces.shape[0] == 1010

    traces, l, r = get_chunk_with_margin(rec_seg, 2000, 3000, None, 10)
    assert l == 10 and r == 10
    assert traces.shape[0] == 1020

    # add zeros
    traces, l, r = get_chunk_with_margin(rec_seg, 5, 1005, None, 10, add_zeros=True)
    assert traces.shape[0] == 1020
    assert l == 10
    assert r == 10
    assert np.all(traces[:5] == 0)
    
    traces, l, r = get_chunk_with_margin(rec_seg, length - 1005, length - 5, None, 10, add_zeros=True)
    assert traces.shape[0] == 1020
    assert np.all(traces[-5:] == 0)
    assert l == 10
    assert r == 10

    traces, l, r = get_chunk_with_margin(rec_seg, length - 500, length + 500, None, 10, add_zeros=True)
    assert traces.shape[0] == 1020
    assert np.all(traces[-510:] == 0)
    assert l == 10
    assert r == 510
    
    # add zeros + window and/or dtype
    traces_windowed, l, r = get_chunk_with_margin(rec_seg, 5, 1005, None, 20, add_zeros=True, window_on_margin=True)
    traces_windowed, l, r = get_chunk_with_margin(rec_seg, length - 1005, length - 5, None, 20, add_zeros=True, window_on_margin=True)
    traces_windowed, l, r = get_chunk_with_margin(rec_seg, length - 500, length + 500, None, 10, add_zeros=True, window_on_margin=True)
    traces, l, r = get_chunk_with_margin(rec_seg, length - 1005, length - 5, None, 20, add_zeros=True, dtype='float64')
    assert traces.dtype == 'float64'

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(traces_windowed[:, 0], color='r', ls='--')
    # plt.show()


def test_order_channels_by_depth():
    rec = generate_recording(num_channels=10, sampling_frequency=1000., durations=[10.], 
                             set_probe=False)
    locations = np.zeros((10, 2))
    locations[:, 1] = np.arange(10) // 2 * 50
    locations[:, 0] = np.arange(10) % 2* 30

    locations_copy = locations.copy()
    locations_copy = locations_copy[::-1]
    rec.set_channel_locations(locations_copy)

    order_1d, order_r1d = order_channels_by_depth(rec, dimensions="y")
    order_2d, order_r2d = order_channels_by_depth(rec, dimensions=("x", "y"))
    locations_rev = locations_copy[order_1d][order_r1d]

    assert np.array_equal(locations[:, 1], locations_copy[order_1d][:, 1])
    assert np.array_equal(locations_copy[order_1d][:, 1], locations_copy[order_2d][:, 1])
    assert np.array_equal(locations, locations_copy[order_2d])
    assert np.array_equal(locations_copy, locations_copy[order_2d][order_r2d])


if __name__ == '__main__':
    # test_get_random_data_chunks()
    # test_get_closest_channels()
    # test_get_noise_levels()
    test_order_channels_by_depth()
