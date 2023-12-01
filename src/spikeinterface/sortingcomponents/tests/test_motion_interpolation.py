import pytest
from pathlib import Path
import numpy as np

from spikeinterface import download_dataset

from spikeinterface.sortingcomponents.motion_interpolation import (
    correct_motion_on_peaks,
    interpolate_motion_on_traces,
    InterpolateMotionRecording,
)

from spikeinterface.sortingcomponents.tests.common import make_dataset


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "sortingcomponents"
else:
    cache_folder = Path("cache_folder") / "sortingcomponents"


def make_fake_motion(rec):
    # make a fake motion vector
    duration = rec.get_total_duration()
    locs = rec.get_channel_locations()
    temporal_bins = np.arange(0.5, duration - 0.49, 0.5)
    spatial_bins = np.arange(locs[:, 1].min(), locs[:, 1].max(), 100)
    motion = np.zeros((temporal_bins.size, spatial_bins.size))
    motion[:, :] = np.linspace(-30, 30, temporal_bins.size)[:, None]

    return motion, temporal_bins, spatial_bins


def test_correct_motion_on_peaks():
    rec, sorting = make_dataset()
    peaks = sorting.to_spike_vector()
    motion, temporal_bins, spatial_bins = make_fake_motion(rec)

    # fake locations
    peak_locations = np.zeros((peaks.size), dtype=[("x", "float32"), ("y", "float")])

    corrected_peak_locations = correct_motion_on_peaks(
        peaks,
        peak_locations,
        rec.sampling_frequency,
        motion,
        temporal_bins,
        spatial_bins,
        direction="y",
    )
    # print(corrected_peak_locations)
    assert np.any(corrected_peak_locations["y"] != 0)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(times[peaks['sample_index']], corrected_peak_locations['y'])
    # ax.plot(temporal_bins, motion[:, 1])
    # plt.show()


def test_interpolate_motion_on_traces():
    rec, sorting = make_dataset()

    motion, temporal_bins, spatial_bins = make_fake_motion(rec)

    channel_locations = rec.get_channel_locations()

    traces = rec.get_traces(segment_index=0, start_frame=0, end_frame=30000)
    times = rec.get_times()[0:30000]

    for method in ("kriging", "idw", "nearest"):
        traces_corrected = interpolate_motion_on_traces(
            traces,
            times,
            channel_locations,
            motion,
            temporal_bins,
            spatial_bins,
            direction=1,
            channel_inds=None,
            spatial_interpolation_method=method,
            spatial_interpolation_kwargs={},
        )
        assert traces.shape == traces_corrected.shape
        assert traces.dtype == traces_corrected.dtype


def test_InterpolateMotionRecording():
    rec, sorting = make_dataset()
    motion, temporal_bins, spatial_bins = make_fake_motion(rec)

    rec2 = InterpolateMotionRecording(rec, motion, temporal_bins, spatial_bins, border_mode="force_extrapolate")
    assert rec2.channel_ids.size == 32

    rec2 = InterpolateMotionRecording(rec, motion, temporal_bins, spatial_bins, border_mode="force_zeros")
    assert rec2.channel_ids.size == 32

    rec2 = InterpolateMotionRecording(rec, motion, temporal_bins, spatial_bins, border_mode="remove_channels")
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
    test_correct_motion_on_peaks()
    test_interpolate_motion_on_traces()
    test_InterpolateMotionRecording()
