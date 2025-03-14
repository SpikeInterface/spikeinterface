import numpy as np
import pytest

from spikeinterface.preprocessing.inter_session_alignment import session_alignment, alignment_utils
from spikeinterface.generation.session_displacement_generator import *
import spikeinterface  # required for monkeypatching
import spikeinterface.full as si


# TODO: this is going to be very slow. Speed it up.
# TODO: finish implementing all NotImplementedError tests
# TODO: tests require pytest mock, will fail without it in a strange way, check explicitly

# TODO: CRITICAL: thoroughly test the motion wrapper, and tidy it up,  it looks jenky as hell!

# TODO: need to handle the force-zeros etc.

# test histogram is accurate (1D and 2D) (TEST MEAN VS. MEDIAN FOR 1D AND 2D
# test chunked histogram is accurate (1D and 2D)
# test all arguments are passed down to the correct subfunction
# test spatial bin centers
# unit test get_2d_activity_histogram extra arguments.
# TODO: test the case where unit locations "y" are outside of the probe!

# The full histogram from chunked histogram is tested in test_histogram_parameters().
# TODO: check median vs mean (simply compute on the chunked_histograms, which are tested here)
# TODO: test log, amplitude scale, etc at high level?

# TODO:
# test the main histogram (variable bin num)
# test the chunked histogram (split the peak locs based on the sample idx!)
# find peaks list within bin_edges
# scale to Hz :)

# 1) test all sub-functions are called with the correct arguments:
# do this by getter

# test histogram generation from known peaks, peak locations
# test temporal histogram bin size (chunked), in general histogram_info
# but test main histogram time window too

"""
TODO: major
- should probably take log after averaging not before.
- fix issue leading to "force_zeros"
- fix bug with motion correction type

TODO: TEST:
 - non-rigid, including num shifts
 - smoothing is applied
 - ask log2 vs. log10 (e.g. for 3d histogram)
 - test the case where unit locations "y" are outside of the probe! (think about if this is really necessary)
 - concat data and see if it imporves sorting. this is more benchmarking...
"""


class TestInterSessionAlignment:
    """ """

    @pytest.fixture(scope="session")
    def test_recording_1(self):
        """
        # if this is say 20, then units go off the edge of the probe and are such low amplitude they are not picked up.
        """
        shifts = ((0, 0), (0, -200), (0, 150))

        recordings_list, _ = generate_session_displacement_recordings(
            num_units=15,
            recording_durations=[9, 10, 11],
            recording_shifts=shifts,
            non_rigid_gradient=None,
            seed=55,  # 52
            generate_sorting_kwargs=dict(firing_rates=(100, 250), refractory_period_ms=4.0),
            generate_unit_locations_kwargs=dict(
                margin_um=0.0,
                minimum_z=0.0,
                maximum_z=2.0,
                minimum_distance=18.0,
                max_iteration=100,
                distance_strict=False,
            ),
            generate_noise_kwargs=dict(noise_levels=(0.0, 0.0), spatial_decay=1.0),
        )

        peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
            recordings_list,
            detect_kwargs={"method": "locally_exclusive"},
            localize_peaks_kwargs={"method": "grid_convolution"},
        )
        return (recordings_list, shifts, peaks_list, peak_locations_list)

    ###########################################################################
    # Functional Tests
    ############################################################################

    # TODO: test shift blocks...
    @pytest.mark.parametrize("histogram_type", ["activity_1d", "activity_2d"])
    @pytest.mark.parametrize("num_shifts_global", [None, 200])
    def test_align_sessions_finds_correct_shifts(self, num_shifts_global, test_recording_1, histogram_type):
        """

        Note - hard coding of expected shifts. 200*bin_um
        """
        recordings_list, shifts, peaks_list, peak_locations_list = test_recording_1

        assert shifts == ((0, 0), (0, -200), (0, 150)), "expected shifts are hard-coded into this test."

        compute_alignment_kwargs = session_alignment.get_compute_alignment_kwargs()
        compute_alignment_kwargs["smoothing_sigma_bin"] = None
        compute_alignment_kwargs["smoothing_sigma_window"] = None
        compute_alignment_kwargs["num_shifts_global"] = num_shifts_global

        estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
        estimate_histogram_kwargs["bin_um"] = 2
        estimate_histogram_kwargs["histogram_type"] = histogram_type
        estimate_histogram_kwargs["log_scale"] = True

        for mode, expected in zip(
            ["to_session_1", "to_session_2", "to_session_3", "to_middle"],
            [
                (0, -200, 150),
                (200, 0, 350),
                (-150, -350, 0),
                (16.66, -183.33, 166.66),
            ],
        ):
            corrected_recordings_list, extra_info = session_alignment.align_sessions(
                recordings_list,
                peaks_list,
                peak_locations_list,
                alignment_order=mode,
                compute_alignment_kwargs=compute_alignment_kwargs,
                estimate_histogram_kwargs=estimate_histogram_kwargs,
            )

            assert np.allclose(expected, extra_info["shifts_array"].squeeze(), rtol=0, atol=0.02)

        corr_peaks_list, corr_peak_loc_list = session_alignment.compute_peaks_locations_for_session_alignment(
            corrected_recordings_list,
            detect_kwargs={"method": "locally_exclusive"},
            localize_peaks_kwargs={"method": "grid_convolution"},
        )

        new_histograms = session_alignment._compute_session_histograms(
            corrected_recordings_list, corr_peaks_list, corr_peak_loc_list, **estimate_histogram_kwargs
        )[0]

        rows, cols = np.triu_indices(len(new_histograms), k=1)
        assert np.all(
            np.abs(np.corrcoef([hist.flatten() for hist in new_histograms])[rows, cols])
            - np.abs(np.corrcoef([hist.flatten() for hist in extra_info["session_histogram_list"]])[rows, cols])
            >= 0
        )

    def test_histogram_generation(self, test_recording_1):
        """ """
        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        recording = recordings_list[0]

        channel_locations = recording.get_channel_locations()
        loc_start = np.min(channel_locations[:, 1])
        loc_end = np.max(channel_locations[:, 1])

        # Test some floats as slightly more complex than integer case
        bin_s = 1.5
        bin_um = 1.5

        estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
        estimate_histogram_kwargs["bin_um"] = bin_um
        estimate_histogram_kwargs["chunked_bin_size_s"] = bin_s
        estimate_histogram_kwargs["log_scale"] = False

        (
            session_histogram_list,
            temporal_bin_centers_list,
            spatial_bin_centers,
            spatial_bin_edges,
            histogram_info_list,
        ) = session_alignment._compute_session_histograms(
            recordings_list, peaks_list, peak_locations_list, **estimate_histogram_kwargs
        )

        num_bins = (loc_end - loc_start) / bin_um

        bin_edges = np.linspace(loc_start, loc_end, int(num_bins) + 1)
        bin_centers = bin_edges[:-1] + bin_um / 2

        assert np.array_equal(bin_edges, spatial_bin_edges)
        assert np.array_equal(bin_centers, spatial_bin_centers)

        for recording, temporal_bin_center in zip(recordings_list, temporal_bin_centers_list):
            times = recording.get_times()
            centers = (np.max(times) - np.min(times)) / 2
            assert temporal_bin_center == centers

        for ses_idx, (recording, chunked_histogram_info) in enumerate(zip(recordings_list, histogram_info_list)):

            # TODO: this is direct copy from above, can merge
            times = recording.get_times()

            chunk_time_window = chunked_histogram_info["chunked_bin_size_s"]

            num_windows = (np.ceil(np.max(times)) - np.min(times)) / chunk_time_window
            temp_bin_edges = np.arange(np.ceil(num_windows) + 1) * chunk_time_window
            centers = temp_bin_edges[:-1] + chunk_time_window / 2

            assert chunked_histogram_info["chunked_bin_size_s"] == chunk_time_window
            assert np.array_equal(chunked_histogram_info["chunked_temporal_bin_centers"], centers)

            for edge_idx in range(len(temp_bin_edges) - 1):

                lower = temp_bin_edges[edge_idx]
                upper = temp_bin_edges[edge_idx + 1]

                lower_idx = recording.time_to_sample_index(lower)
                upper_idx = recording.time_to_sample_index(upper)

                new_peak_locs = peak_locations_list[ses_idx][
                    np.where(
                        np.logical_and(
                            peaks_list[ses_idx]["sample_index"] >= lower_idx,
                            peaks_list[ses_idx]["sample_index"] < upper_idx,
                        )
                    )
                ]
                assert np.allclose(
                    np.histogram(new_peak_locs["y"], bins=bin_edges)[0] / (upper - lower),
                    chunked_histogram_info["chunked_histograms"][edge_idx, :],
                    rtol=0,
                    atol=1e-6,
                )

    @pytest.mark.parametrize("histogram_type", ["activity_1d", "activity_2d"])
    @pytest.mark.parametrize("operator", ["mean", "median"])
    def test_histogram_parameters(self, test_recording_1, histogram_type, operator):
        """ """
        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
        estimate_histogram_kwargs["log_scale"] = False
        estimate_histogram_kwargs["method"] = f"chunked_{operator}"
        estimate_histogram_kwargs["histogram_type"] = histogram_type

        _, extra_info = session_alignment.align_sessions(
            recordings_list, peaks_list, peak_locations_list, estimate_histogram_kwargs=estimate_histogram_kwargs
        )
        estimate_histogram_kwargs["log_scale"] = True

        _, extra_info_log = session_alignment.align_sessions(
            recordings_list, peaks_list, peak_locations_list, estimate_histogram_kwargs=estimate_histogram_kwargs
        )
        for ses_idx in range(len(recordings_list)):

            ses_hist = extra_info["session_histogram_list"][ses_idx]

            chunked_histograms = extra_info["histogram_info_list"][ses_idx]["chunked_histograms"]
            for chunk_hist, chunk_hist_log in zip(
                chunked_histograms, extra_info_log["histogram_info_list"][ses_idx]["chunked_histograms"]
            ):
                assert np.array_equal(np.log2(chunk_hist + 1), chunk_hist_log)

            summary_func = np.median if operator == "median" else np.mean
            assert np.array_equal(ses_hist, summary_func(chunked_histograms, axis=0))

    ###########################################################################
    # Kwargs Tests
    ###########################################################################

    #             "histogram_type": "activity_1d",  # _get_single_session_activity_histogram
    # "log_scale": True,  # get_2d_activity_histogram
    # "bin_um": 5,  # make_2d_motion_histogram
    # "method": "chunked_median",  # _get_single_session_activity_histogram

    def test_get_estimate_histogram_kwargs(self, mocker, test_recording_1):
        #  breakpoint()
        # to make_2d_motion_histogram

        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        default_kwargs = {
            "bin_um": 2,
            "method": "chunked_mean",
            "chunked_bin_size_s": "estimate",
            "log_scale": True,
            "depth_smooth_um": None,
            "histogram_type": "activity_1d",
            "weight_with_amplitude": False,
            "avg_in_bin": False,
        }

        assert (
            default_kwargs == session_alignment.get_estimate_histogram_kwargs()
        ), "Default `get_estimate_histogram_kwargs` were changed."

        different_kwargs = session_alignment.get_estimate_histogram_kwargs()
        different_kwargs.update(
            {
                "chunked_bin_size_s": 6,
                "depth_smooth_um": 5,
                "weight_with_amplitude": True,
                "avg_in_bin": True,
            }
        )

        spy_2d_histogram = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.alignment_utils, "make_2d_motion_histogram"
        )
        session_alignment.align_sessions(
            [recordings_list[0]], [peaks_list[0]], [peak_locations_list[0]], estimate_histogram_kwargs=different_kwargs
        )
        first_call = spy_2d_histogram.call_args_list[0]
        args, kwargs = first_call

        assert kwargs["bin_s"] == different_kwargs["chunked_bin_size_s"]
        assert kwargs["bin_um"] is None
        assert np.unique(np.diff(kwargs["spatial_bin_edges"])) == different_kwargs["bin_um"]
        assert kwargs["depth_smooth_um"] == different_kwargs["depth_smooth_um"]
        assert kwargs["weight_with_amplitude"] == different_kwargs["weight_with_amplitude"]
        assert kwargs["avg_in_bin"] == different_kwargs["avg_in_bin"]

    # akima_interp_nonrigid
    # assert kwargs["num_shifts"] == different_kwargs["num_shifts_global"]
    # assert kwargs["interpolate"] == different_kwargs["interpolate"]
    # assert kwargs["interp_factor"] == different_kwargs["interp_factor"]
    def test_compute_alignment_kwargs(self, mocker, test_recording_1):

        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        default_kwargs = {
            "num_shifts_global": None,
            "num_shifts_block": 20,
            "interpolate": False,
            "interp_factor": 10,
            "kriging_sigma": 1,
            "kriging_p": 2,
            "kriging_d": 2,
            "smoothing_sigma_bin": 0.5,
            "smoothing_sigma_window": 0.5,
            "akima_interp_nonrigid": False,
        }
        assert (
            session_alignment.get_compute_alignment_kwargs() == default_kwargs
        ), "Default `get_compute_alignment_kwargs` were changed."

        different_kwargs = session_alignment.get_compute_alignment_kwargs()
        different_kwargs.update(
            {
                "interpolate": True,
                "kriging_sigma": 5,
                "kriging_p": 10,
                "kriging_d": 20,
                "smoothing_sigma_bin": 1.2,
                "smoothing_sigma_window": 1.3,
            }
        )
        spy_kriging = mocker.spy(spikeinterface.preprocessing.inter_session_alignment.alignment_utils, "kriging_kernel")
        spy_gaussian_filter = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.alignment_utils, "gaussian_filter"
        )
        non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
        non_rigid_window_kwargs["rigid"] = False

        session_alignment.align_sessions(
            [recordings_list[0]],
            [peaks_list[0]],
            [peak_locations_list[0]],
            compute_alignment_kwargs=different_kwargs,
            non_rigid_window_kwargs=non_rigid_window_kwargs,
        )
        kwargs = spy_kriging.call_args_list[0][1]
        assert kwargs["sigma"] == different_kwargs["kriging_sigma"]
        assert kwargs["p"] == different_kwargs["kriging_p"]
        assert kwargs["d"] == different_kwargs["kriging_d"]

        # First call is overall rigid, then nonrigid smooth bin
        kwargs = spy_gaussian_filter.call_args_list[0][1]
        assert kwargs["sigma"] == different_kwargs["smoothing_sigma_bin"]

        # then nonrigid smooth window
        kwargs = spy_gaussian_filter.call_args_list[2][1]
        assert kwargs["sigma"] == different_kwargs["smoothing_sigma_window"]

    def test_non_rigid_window_kwargs(self, mocker, test_recording_1):

        import spikeinterface  # TODO: place up top with a note

        default_kwargs = {
            "rigid": True,
            "win_shape": "gaussian",
            "win_step_um": 50,
            "win_scale_um": 50,
            "win_margin_um": None,
            "zero_threshold": None,
        }
        assert (
            session_alignment.get_non_rigid_window_kwargs() == default_kwargs
        ), "Default `get_non_rigid_window_kwargs` were changed."

        different_kwargs = {
            "rigid": False,
            "win_shape": "rect",
            "win_step_um": 55,
            "win_scale_um": 65,
            "win_margin_um": 10,
            "zero_threshold": 4,
        }

        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        spy_get_spatial_windows = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.session_alignment, "get_spatial_windows"
        )
        session_alignment.align_sessions(
            [recordings_list[0]], [peaks_list[0]], [peak_locations_list[0]], non_rigid_window_kwargs=different_kwargs
        )
        kwargs = spy_get_spatial_windows.call_args_list[0][1]
        assert kwargs["rigid"] == different_kwargs["rigid"]
        assert kwargs["win_shape"] == different_kwargs["win_shape"]
        assert kwargs["win_step_um"] == different_kwargs["win_step_um"]
        assert kwargs["win_scale_um"] == different_kwargs["win_scale_um"]
        assert kwargs["win_margin_um"] == different_kwargs["win_margin_um"]
        assert kwargs["zero_threshold"] == different_kwargs["zero_threshold"]

    def test_interpolate_motion_kwargs(self, mocker, test_recording_1):
        # InterpolateMotionRecording <- check times on this created object

        default_kwargs = {
            "border_mode": "force_zeros",  # fixed as this until can figure out probe
            "spatial_interpolation_method": "kriging",
            "sigma_um": 20.0,
            "p": 2,
        }
        assert (
            session_alignment.get_interpolate_motion_kwargs() == default_kwargs
        ), "Default `get_non_rigid_window_kwargs` were changed."

        different_kwargs = {
            "border_mode": "force_zeros",  # fixed as this until can figure out probe
            "spatial_interpolation_method": "nearest",
            "sigma_um": 25.0,
            "p": 3,
        }

        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        spy_get_2d_activity_histogram = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.session_alignment.InterpolateMotionRecording,
            "__init__",
        )
        session_alignment.align_sessions(
            [recordings_list[0]], [peaks_list[0]], [peak_locations_list[0]], interpolate_motion_kwargs=different_kwargs
        )
        first_call = spy_get_2d_activity_histogram.call_args_list[0]
        args, kwargs = first_call

        assert kwargs["border_mode"] == different_kwargs["border_mode"]
        assert kwargs["spatial_interpolation_method"] == different_kwargs["spatial_interpolation_method"]
        assert kwargs["sigma_um"] == different_kwargs["sigma_um"]
        assert kwargs["p"] == different_kwargs["p"]

    ###########################################################################
    # Following Motion Correction
    ###########################################################################

    def test_rigid_motion_rigid_intersession(self):
        """ """
        mc_recording_list, mc_motion_info_list, shifts = self.get_motion_corrected_recordings_list(
            rigid_motion=True, rigid_intersession=True
        )

        # should assert these are zero!
        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        assert first_ses_mc_displacement[0].size == 1
        assert second_ses_mc_displacement[0].size == 1

        first_ses_mc_displacement[0] += 0.01
        second_ses_mc_displacement[0] += 0.02

        non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
        non_rigid_window_kwargs["rigid"] = True

        corrected_recordings, extra_info = session_alignment.align_sessions_after_motion_correction(
            mc_recording_list,
            mc_motion_info_list,
            align_sessions_kwargs={
                "alignment_order": "to_session_1",
                "non_rigid_window_kwargs": non_rigid_window_kwargs,
            },
        )
        first_ses_total_displacement = corrected_recordings[0]._recording_segments[0].motion.displacement
        second_ses_total_displacement = corrected_recordings[1]._recording_segments[0].motion.displacement

        assert first_ses_total_displacement == [np.array([[0.01]])]  # TODO: can use shifts directly!
        assert second_ses_total_displacement == [np.array([[250.02]])]

    def test_rigid_motion_nonrigid_intersession(self):

        # TODO: DIRECT COPY!
        mc_recording_list, mc_motion_info_list, shifts = self.get_motion_corrected_recordings_list(
            rigid_motion=True, rigid_intersession=False
        )
        # should assert these are zero!
        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        assert first_ses_mc_displacement[0].size == 1
        assert second_ses_mc_displacement[0].size == 1

        first_ses_mc_displacement[0] += 0.01
        second_ses_mc_displacement[0] += 0.02

        # All of this is direct copy between these tests...
        non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
        non_rigid_window_kwargs["rigid"] = False

        # TODO: can we take this mc_motion_info_list directly from the recordings? double check...
        corrected_recordings, extra_info = session_alignment.align_sessions_after_motion_correction(
            mc_recording_list,
            mc_motion_info_list,
            align_sessions_kwargs={
                "alignment_order": "to_session_1",
                "non_rigid_window_kwargs": non_rigid_window_kwargs,
            },
        )

        first_ses_total_displacement = corrected_recordings[0]._recording_segments[0].motion.displacement
        second_ses_total_displacement = corrected_recordings[1]._recording_segments[0].motion.displacement

        assert np.all(extra_info["shifts_array"][0] + 0.01 == first_ses_total_displacement)
        # TODO: why are these nonlinear shifts so wrong?
        assert np.all(extra_info["shifts_array"][1] + 0.02 == second_ses_total_displacement)

    # TODO: Does this really test the passed peak list, peak_locations list? maybe use a monkeypatch?
    # TODO: monkeypatch through all passed args?
    @pytest.mark.parametrize("rigid_intersession", [True, False])
    def test_nonrigid_motion(self, rigid_intersession):
        # TODO: DIRECT COPY!
        mc_recording_list, mc_motion_info_list, shifts = self.get_motion_corrected_recordings_list(
            rigid_motion=False, rigid_intersession=rigid_intersession
        )
        # should assert these are zero!
        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        offsets1 = np.linspace(0, 0.1, first_ses_mc_displacement[0].size)
        offsets2 = np.linspace(0, 0.1, first_ses_mc_displacement[0].size)

        first_ses_mc_displacement[0] += offsets1
        second_ses_mc_displacement[0] += offsets2

        # All of this is direct copy between these tests...

        non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
        non_rigid_window_kwargs["rigid"] = rigid_intersession

        corrected_recordings, extra_info = (
            session_alignment.align_sessions_after_motion_correction(  # TODO: check it is interepolate motion recording, tell them to use other function if it is !
                mc_recording_list,
                mc_motion_info_list,
                align_sessions_kwargs={
                    "alignment_order": "to_session_1",
                    "non_rigid_window_kwargs": non_rigid_window_kwargs,
                },
            )
        )

        first_ses_total_displacement = corrected_recordings[0]._recording_segments[0].motion.displacement
        second_ses_total_displacement = corrected_recordings[1]._recording_segments[0].motion.displacement

        assert np.all(extra_info["shifts_array"][0] + offsets1 == first_ses_total_displacement)
        assert np.all(extra_info["shifts_array"][1] + offsets2 == second_ses_total_displacement)

    def get_motion_corrected_recordings_list(self, rigid_motion=True, rigid_intersession=True):
        # TODO: these kwargs are copied from abvove. I guess this can't be a fixure because of the arguments.

        shifts = ((0, 0), (0, 250))
        non_rigid_gradient = None if rigid_intersession else 0.2
        recordings_list, _ = generate_session_displacement_recordings(
            num_units=5,
            recording_durations=[1, 1],
            recording_shifts=shifts,
            non_rigid_gradient=non_rigid_gradient,
            seed=55,  # 52
            generate_sorting_kwargs=dict(firing_rates=(100, 250), refractory_period_ms=4.0),
            generate_unit_locations_kwargs=dict(
                margin_um=0.0,
                minimum_z=0.0,
                maximum_z=2.0,
                minimum_distance=18.0,
                max_iteration=100,
                distance_strict=False,
            ),
            generate_noise_kwargs=dict(noise_levels=(0.0, 0.5), spatial_decay=1.0),
            # must have some noise, or peak detection becomes completely stoachastic!
        )

        interpolate_motion_kwargs = {
            "border_mode": "force_zeros"
        }  # TODO: unfortunately this is necessary for now until fixed
        localize_peaks_kwargs = {"method": "grid_convolution"}

        preset = "rigid_fast" if rigid_motion else "kilosort_like"

        mc_recording_list = []
        mc_motion_info_list = []
        for rec in recordings_list:
            corrected_rec, motion_info = si.correct_motion(
                rec,
                preset=preset,
                interpolate_motion_kwargs=interpolate_motion_kwargs,
                output_motion_info=True,
                localize_peaks_kwargs=localize_peaks_kwargs,
            )
            mc_recording_list.append(corrected_rec)
            mc_motion_info_list.append(motion_info)

        return mc_recording_list, mc_motion_info_list, shifts  #

    ###########################################################################
    # Unit Tests
    ###########################################################################

    def test_shift_array_fill_zeros(self):
        """ """
        # TODO: could do all the same indexing as 1d? but will be more confusing...

        # Test 1d array
        test_1d = np.random.random((10))

        # shift leftwards
        res = alignment_utils.shift_array_fill_zeros(test_1d, 2)
        assert np.all(res[8:] == 0)
        assert np.array_equal(res[:8], test_1d[2:])

        # shift rightwards
        res = alignment_utils.shift_array_fill_zeros(test_1d, -2)
        assert np.all(res[:2] == 0)
        assert np.array_equal(res[2:], test_1d[:8])

        # Test 2d array
        test_2d = np.random.random((10, 10))

        # shift upwards
        res = alignment_utils.shift_array_fill_zeros(test_2d, 2)
        assert np.all(res[8:, :] == 0)
        assert np.array_equal(res[:8, :], test_2d[2:, :])

        # shift downwards.
        res = alignment_utils.shift_array_fill_zeros(test_2d, -2)
        assert np.all(res[:2, :] == 0)
        assert np.array_equal(res[2:, :], test_2d[:8, :])

    def test_get_shifts_from_session_matrix(self):
        """ """
        matrix = np.random.random((10, 10, 2))

        res = alignment_utils.get_shifts_from_session_matrix("to_middle", matrix)
        assert np.array_equal(res, -np.mean(matrix, axis=0))

        res = alignment_utils.get_shifts_from_session_matrix("to_session_1", matrix)
        assert np.array_equal(res, -matrix[0, :, :])

        res = alignment_utils.get_shifts_from_session_matrix("to_session_5", matrix)
        assert np.array_equal(res, -matrix[4, :, :])

        res = alignment_utils.get_shifts_from_session_matrix("to_session_10", matrix)
        assert np.array_equal(res, -matrix[9, :, :])

    @pytest.mark.parametrize("interpolate", [True, False])
    @pytest.mark.parametrize("odd_hist_size", [True, False])
    @pytest.mark.parametrize("shifts", [3, -2])
    # TODO: return and test num_shifts
    # TODO: test
    def test_compute_histogram_crosscorrelation(self, interpolate, odd_hist_size, shifts):

        if odd_hist_size:
            hist = np.array([1, 0, 1, 1, 1, 0])
        else:
            hist = np.array([0, 0, 1, 1, 0, 1, 0, 1])

        hist_shift = alignment_utils.shift_array_fill_zeros(hist, shifts)

        session_histogram_list = np.vstack([hist, hist_shift])

        interp_factor = 50  # not used when interpolate = False
        shifts_matrix, xcorr_matrix_unsmoothed = alignment_utils.compute_histogram_crosscorrelation(
            session_histogram_list,
            non_rigid_windows=np.ones((1, hist.size)),
            num_shifts=None,
            interpolate=interpolate,
            interp_factor=interp_factor,
            kriging_sigma=0.2,
            kriging_p=2,
            kriging_d=2,
            smoothing_sigma_bin=None,
            smoothing_sigma_window=None,
        )
        assert np.isclose(
            alignment_utils.get_shifts_from_session_matrix("to_session_1", shifts_matrix)[-1],
            -shifts,
            rtol=0,
            atol=0.01,
        )

        num_shifts = hist.size * 2 - 1
        if interpolate:
            assert xcorr_matrix_unsmoothed.shape[1] == num_shifts * interp_factor
        else:
            assert xcorr_matrix_unsmoothed.shape[1] == num_shifts

    def test_compute_histogram_crosscorrelation_gaussian_filter_kwargs(self):
        pass

    def estimate_chunk_size(self):
        pass

    def test_akima_interpolate_nonrigid_shifts(self):
        pass

    ###########################################################################
    # Benchmarking
    ###########################################################################

    # badly need to review repeat from motion correction alg

    # 1) create some inter-session drift
    # 2) compare sorting before / after
    # 3) need to benchmark fully


# test_compute_alignment_kwargs
