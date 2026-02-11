import warnings

import numpy as np
import pickle

from spikeinterface import NumpyRecording, load
from spikeinterface.core import generate_ground_truth_recording
from spikeinterface.core.motion import Motion
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.sortingcomponents.motion.motion_interpolation import (
    InterpolateMotionRecording,
    correct_motion_on_peaks,
    interpolate_motion,
    interpolate_motion_on_traces,
)
from spikeinterface.sortingcomponents.tests.common import make_dataset


def make_fake_motion(rec):
    # make a fake motion object

    locs = rec.get_channel_locations()
    spatial_bins = np.arange(locs[:, 1].min(), locs[:, 1].max(), 100)

    displacement = []
    temporal_bins = []
    for segment_index in range(rec.get_num_segments()):
        duration = rec.get_duration(segment_index=segment_index)
        seg_time_bins = np.arange(0.5, duration - 0.49, 0.5)
        seg_disp = np.zeros((seg_time_bins.size, spatial_bins.size))
        seg_disp[:, :] = np.linspace(-30, 30, seg_time_bins.size)[:, None]

        temporal_bins.append(seg_time_bins)
        displacement.append(seg_disp)

    motion = Motion(displacement, temporal_bins, spatial_bins, direction="y")

    return motion


def test_motion_object_interpolators(tmp_path):
    rec, sorting = make_dataset()
    motion = make_fake_motion(rec)

    # serialize with pickle after interpolation fit
    motion.make_interpolators()
    assert motion.interpolators is not None
    motion2 = pickle.loads(pickle.dumps(motion))
    assert motion2.interpolators is not None

    # to/from dict
    motion2 = Motion.from_dict(motion.to_dict())
    assert motion == motion2
    # serialization to/from dict loses interpolators
    assert motion2.interpolators is None
    motion2.make_interpolators()

    # save/load to folder
    folder = tmp_path / "motion_saved"
    motion.save(folder)
    motion3 = Motion.load(folder)
    # save/load to folder loses interpolators
    assert motion3.interpolators is None

    # do interpolate
    displacement = motion.get_displacement_at_time_and_depth([2, 4.4, 11], [120.0, 80.0, 150.0])
    displacement2 = motion2.get_displacement_at_time_and_depth([2, 4.4, 11], [120.0, 80.0, 150.0])
    displacement3 = motion3.get_displacement_at_time_and_depth([2, 4.4, 11], [120.0, 80.0, 150.0])
    # print(displacement)
    assert displacement.shape[0] == 3
    # check interpolators reload
    np.testing.assert_array_equal(displacement, displacement2)
    np.testing.assert_array_equal(displacement, displacement3)

    # interpolate grid
    displacement = motion.get_displacement_at_time_and_depth([2, 4.4, 11, 15, 19], [150.0, 80.0], grid=True)
    assert displacement.shape == (2, 5)


def test_correct_motion_on_peaks():
    rec, sorting = make_dataset()
    peaks = sorting.to_spike_vector()
    motion = make_fake_motion(rec)

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
        for interpolation_time_bin_centers_s in (None, np.linspace(*times[[0, -1]], num=3)):
            traces_corrected = interpolate_motion_on_traces(
                traces,
                times,
                channel_locations,
                motion,
                channel_inds=None,
                spatial_interpolation_method=method,
                interpolation_time_bin_centers_s=interpolation_time_bin_centers_s,
                # spatial_interpolation_kwargs={},
                spatial_interpolation_kwargs={"force_extrapolate": True},
            )
            assert traces.shape == traces_corrected.shape
            assert traces.dtype == traces_corrected.dtype


def test_interpolation_simple():
    # a recording where a 1 moves at 1 chan per second. 30 chans 10 frames.
    # there will be 9 chans of drift, so we add 9 chans of padding to the bottom
    n_samples = num_chans_orig = 10  # these need to be the same for this test
    num_chans_drifted = num_chans_orig + num_chans_orig - 1
    traces = np.zeros((n_samples, num_chans_drifted), dtype="float32")
    traces[:, :num_chans_orig] = np.eye(num_chans_orig)
    rec = NumpyRecording(traces, sampling_frequency=1)
    rec.set_dummy_probe_from_locations(np.c_[np.zeros(num_chans_drifted), np.arange(num_chans_drifted)])

    true_motion = Motion(np.arange(n_samples)[:, None], 0.5 + np.arange(n_samples), np.zeros(1))
    rec_corrected = interpolate_motion(rec, true_motion, spatial_interpolation_method="nearest")
    traces_corrected = rec_corrected.get_traces()
    assert traces_corrected.shape == (num_chans_orig, num_chans_orig)
    assert np.array_equal(traces_corrected[:, 0], np.ones(n_samples))
    assert np.array_equal(traces_corrected[:, 1:], np.zeros((n_samples, num_chans_orig - 1)))

    # let's try a new version where we interpolate too slowly
    rec_corrected = interpolate_motion(
        rec, true_motion, spatial_interpolation_method="nearest", num_closest=2, interpolation_time_bin_size_s=2
    )
    traces_corrected = rec_corrected.get_traces()
    assert traces_corrected.shape == (num_chans_orig, num_chans_orig)
    # what happens with nearest here?
    # well... due to rounding towards the nearest even number, the motion (which at
    # these time bin centers is 0.5, 2.5, 4.5, ...) flips the signal's nearest
    # neighbor back and forth between the first and second channels
    assert np.all(traces_corrected[::2, 0] == 1)
    assert np.all(traces_corrected[1::2, 0] == 0)
    assert np.all(traces_corrected[1::2, 1] == 1)
    assert np.all(traces_corrected[::2, 1] == 0)
    assert np.all(traces_corrected[:, 2:] == 0)


def test_cross_band_interpolation():
    """Simple version of using LFP to interpolate AP data

    This also tests the time vector implementation in interpolation.
    The idea is to have two recordings which are all 0s with a 1 that
    moves from one channel to another after 3s. They're at different
    sampling frequencies. motion estimation in one sampling frequency
    applied to the other should still lead to perfect correction.
    """
    from spikeinterface.sortingcomponents.motion import estimate_motion

    # sampling freqs and timing for AP and LFP recordings
    fs_lfp = 50.0
    fs_ap = 300.0
    t_start = 10.0
    total_duration = 5.0
    num_samples_lfp = int(fs_lfp * total_duration)
    num_samples_ap = int(fs_ap * total_duration)
    t_switch = 3

    # because interpolation uses bin centers logic, there will be a half
    # bin offset at the change point in the AP recording.
    halfbin_ap_lfp = int(0.5 * (fs_ap / fs_lfp))

    # channel geometry
    num_chans = 10
    geom = np.c_[np.zeros(num_chans), np.arange(num_chans)]

    # make an LFP recording which drifts a bit
    traces_lfp = np.zeros((num_samples_lfp, num_chans))
    traces_lfp[: int(t_switch * fs_lfp), 5] = 1.0
    traces_lfp[int(t_switch * fs_lfp) :, 6] = 1.0
    rec_lfp = NumpyRecording(traces_lfp, sampling_frequency=fs_lfp)
    rec_lfp.set_dummy_probe_from_locations(geom)

    # same for AP
    traces_ap = np.zeros((num_samples_ap, num_chans))
    traces_ap[: int(t_switch * fs_ap) - halfbin_ap_lfp, 5] = 1.0
    traces_ap[int(t_switch * fs_ap) - halfbin_ap_lfp :, 6] = 1.0
    rec_ap = NumpyRecording(traces_ap, sampling_frequency=fs_ap)
    rec_ap.set_dummy_probe_from_locations(geom)

    # set times for both, and silence the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rec_lfp.set_times(t_start + np.arange(num_samples_lfp) / fs_lfp)
        rec_ap.set_times(t_start + np.arange(num_samples_ap) / fs_ap)

    # estimate motion
    motion = estimate_motion(rec_lfp, method="dredge_lfp", rigid=True)

    # nearest to keep it simple
    rec_corrected = interpolate_motion(rec_ap, motion, spatial_interpolation_method="nearest", num_closest=2)
    traces_corrected = rec_corrected.get_traces()
    target = np.zeros((num_samples_ap, num_chans - 2))
    target[:, 4] = 1
    ii, jj = np.nonzero(traces_corrected)
    assert np.array_equal(traces_corrected, target)


def test_InterpolateMotionRecording():
    # rec, sorting = make_dataset()

    # 2 segments
    rec, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=30000.0,
        num_channels=32,
        num_units=10,
        generate_probe_kwargs=dict(
            num_columns=2,
            xpitch=20,
            ypitch=20,
            contact_shapes="circle",
            contact_shape_params={"radius": 6},
        ),
        generate_sorting_kwargs=dict(firing_rates=6.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )

    motion = make_fake_motion(rec)

    rec2 = InterpolateMotionRecording(rec, motion, border_mode="force_extrapolate")
    assert rec2.channel_ids.size == 32

    rec2 = InterpolateMotionRecording(rec, motion, border_mode="force_zeros")
    assert rec2.channel_ids.size == 32

    rec2 = InterpolateMotionRecording(rec, motion, border_mode="remove_channels")
    assert rec2.channel_ids.size == 24
    for ch_id in ("0", "1", "14", "15", "16", "17", "30", "31"):
        assert ch_id not in rec2.channel_ids

    traces = rec2.get_traces(segment_index=0, start_frame=0, end_frame=30000)
    assert traces.shape == (30000, 24)

    traces = rec2.get_traces(segment_index=0, start_frame=0, end_frame=30000, channel_ids=["3", "4"])
    assert traces.shape == (30000, 2)

    # test dump.load when multi segments
    rec2.dump("rec_motion_interp.pickle")
    rec3 = load("rec_motion_interp.pickle")

    check_recordings_equal(rec2, rec3)

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
    # test_interpolation_simple()
    test_InterpolateMotionRecording()
    # test_cross_band_interpolation()
