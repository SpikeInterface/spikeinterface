from pathlib import Path

import numpy as np
import pytest
import spikeinterface.core as sc
from spikeinterface import download_dataset
from spikeinterface.sortingcomponents.motion.motion_interpolation import (
    InterpolateMotionRecording,
    correct_motion_on_peaks,
    interpolate_motion,
    interpolate_motion_on_traces,
)
from spikeinterface.sortingcomponents.motion import Motion
from spikeinterface.sortingcomponents.tests.common import make_dataset


def make_fake_motion(rec):
    # make a fake motion object
    duration = rec.get_total_duration()
    locs = rec.get_channel_locations()
    temporal_bins = np.arange(0.5, duration - 0.49, 0.5)
    spatial_bins = np.arange(locs[:, 1].min(), locs[:, 1].max(), 100)
    displacement = np.zeros((temporal_bins.size, spatial_bins.size))
    displacement[:, :] = np.linspace(-30, 30, temporal_bins.size)[:, None]

    motion = Motion([displacement], [temporal_bins], spatial_bins, direction="y")

    return motion


def test_correct_motion_on_peaks():
    rec, sorting = make_dataset()
    peaks = sorting.to_spike_vector()
    print(peaks.dtype)
    motion = make_fake_motion(rec)
    # print(motion)

    # fake locations
    peak_locations = np.zeros((peaks.size), dtype=[("x", "float32"), ("y", "float")])

    corrected_peak_locations = correct_motion_on_peaks(
        peaks,
        peak_locations,
        motion,
        rec,
    )
    # print(corrected_peak_locations)
    assert np.any(corrected_peak_locations["y"] != 0)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # segment_index = 0
    # times = rec.get_times(segment_index=segment_index)
    # ax.scatter(times[peaks['sample_index']], corrected_peak_locations['y'])
    # ax.plot(motion.temporal_bins_s[segment_index],  motion.displacement[segment_index][:, 1])
    # plt.show()


def test_interpolate_motion_on_traces():
    rec, sorting = make_dataset()

    motion = make_fake_motion(rec)

    channel_locations = rec.get_channel_locations()

    traces = rec.get_traces(segment_index=0, start_frame=0, end_frame=30000)
    times = rec.get_times()[0:30000]

    for method in ("kriging", "idw", "nearest"):
        traces_corrected = interpolate_motion_on_traces(
            traces,
            times,
            channel_locations,
            motion,
            channel_inds=None,
            spatial_interpolation_method=method,
            # spatial_interpolation_kwargs={},
            spatial_interpolation_kwargs={"force_extrapolate": True},
        )
        assert traces.shape == traces_corrected.shape
        assert traces.dtype == traces_corrected.dtype


def test_interpolation_simple():
    # a recording where a 1 moves at 1 chan per second. 30 chans 10 frames.
    # there will be 9 chans of drift, so we add 9 chans of padding to the bottom
    nt = nc0 = 10  # these need to be the same for this test
    nc1 = nc0 + nc0 - 1
    traces = np.zeros((nt, nc1), dtype="float32")
    traces[:, :nc0] = np.eye(nc0)
    rec = sc.NumpyRecording(traces, sampling_frequency=1)
    rec.set_dummy_probe_from_locations(np.c_[np.zeros(nc1), np.arange(nc1)])

    true_motion = Motion(np.arange(nt)[:, None], 0.5 + np.arange(nt), np.zeros(1))
    rec_corrected = interpolate_motion(rec, true_motion, spatial_interpolation_method="nearest")
    traces_corrected = rec_corrected.get_traces()
    assert traces_corrected.shape == (nc0, nc0)
    assert np.array_equal(traces_corrected[:, 0], np.ones(nt))
    assert np.array_equal(traces_corrected[:, 1:], np.zeros((nt, nc0 - 1)))

    # let's try a new version where we interpolate too slowly
    rec_corrected = interpolate_motion(
        rec, true_motion, spatial_interpolation_method="nearest", num_closest=2, interpolation_time_bin_size_s=2
    )
    traces_corrected = rec_corrected.get_traces()
    assert traces_corrected.shape == (nc0, nc0)
    # what happens with nearest here?
    # well... due to rounding towards the nearest even number, the motion (which at
    # these time bin centers is 0.5, 2.5, 4.5, ...) flips the signal's nearest
    # neighbor back and forth between the first and second channels
    assert np.all(traces_corrected[::2, 0] == 1)
    assert np.all(traces_corrected[1::2, 0] == 0)
    assert np.all(traces_corrected[1::2, 1] == 1)
    assert np.all(traces_corrected[::2, 1] == 0)
    assert np.all(traces_corrected[:, 2:] == 0)


def test_InterpolateMotionRecording():
    rec, sorting = make_dataset()
    motion = make_fake_motion(rec)

    rec2 = InterpolateMotionRecording(rec, motion, border_mode="force_extrapolate")
    assert rec2.channel_ids.size == 32

    rec2 = InterpolateMotionRecording(rec, motion, border_mode="force_zeros")
    assert rec2.channel_ids.size == 32

    rec2 = InterpolateMotionRecording(rec, motion, border_mode="remove_channels")
    assert rec2.channel_ids.size == 24
    for ch_id in (0, 1, 14, 15, 16, 17, 30, 31):
        assert ch_id not in rec2.channel_ids

    traces = rec2.get_traces(segment_index=0, start_frame=0, end_frame=30000)
    assert traces.shape == (30000, 24)

    traces = rec2.get_traces(segment_index=0, start_frame=0, end_frame=30000, channel_ids=[3, 4])
    assert traces.shape == (30000, 2)

    # import matplotlib.pyplot as plt
    # import spikeinterface.widgets as sw
    # fig, ax = plt.subplots()
    # sw.plot_probe_map(rec, with_channel_ids=True, ax=ax)
    # fig, ax = plt.subplots()
    # sw.plot_probe_map(rec2, with_channel_ids=True, ax=ax)
    # plt.show()


if __name__ == "__main__":
    # test_correct_motion_on_peaks()
    # test_interpolate_motion_on_traces()
    test_interpolation_simple()
    test_InterpolateMotionRecording()
