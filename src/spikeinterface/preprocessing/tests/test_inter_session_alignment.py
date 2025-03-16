import numpy as np
import pytest

from spikeinterface.preprocessing.inter_session_alignment import session_alignment, alignment_utils
from spikeinterface.generation.session_displacement_generator import *
import spikeinterface  # required for monkeypatching
import spikeinterface.full as si
from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

DEBUG = False


"""
TODO: major
- should probably take log after averaging not before.

TODO: TEST:
 - 'get_traces' is never tested here. Can try once after motion?
 - smoothing is applied

 - ask log2 vs. log10 (e.g. for 3d histogram)
 - ask charlie about threshold
"""


class TestInterSessionAlignment:

    @pytest.fixture(scope="session")
    def recording_1(self):
        """
        Generate a set of session recordings with displacement.
        These parameters are chosen such that simulated AP signal is strong
        on the probe to avoid  noise in the AP positions. This is important
        for checking that the estimated shift matches the known shift.
        """
        shifts = ((0, 0), (0, -200), (0, 150))

        recordings_list, _ = generate_session_displacement_recordings(
            num_units=15,
            recording_durations=[0.1, 0.2, 0.3],
            recording_shifts=shifts,
            non_rigid_gradient=None,
            seed=55,
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

    # TODO: need to make this a fixtures somehow, I guess rigid or nonrigid case.
    @pytest.fixture(scope="session")
    def recording_2(self):
        """
        Get a shifted inter-session alignment recording. First, interpolate-motion
        within session. The purpose of these tests is then to run inter-session alignment
        and check the displacements are properly added.
        """
        shifts = ((0, 0), (0, 250))

        recordings_list, _ = generate_session_displacement_recordings(
            num_units=5,
            recording_durations=[0.3, 0.3],
            recording_shifts=shifts,
            non_rigid_gradient=0.2,
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
            # must have some noise, or peak detection becomes completely stoachastic
            # because it relies on the std to set the threshold.
        )

        peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
            recordings_list,
            detect_kwargs={"method": "locally_exclusive"},
            localize_peaks_kwargs={"method": "grid_convolution"},
        )

        return (recordings_list, shifts, peaks_list, peak_locations_list)

    def motion_correct_recordings_list(self, recordings_list, rigid_motion):
        # Unfortunately this is necessary only in the test environemnt
        # because adding offsets means copying the motion object does not work
        interpolate_motion_kwargs = {"border_mode": "force_zeros"}
        localize_peaks_kwargs = {"method": "grid_convolution"}

        preset = "rigid_fast" if rigid_motion else "kilosort_like"

        # Perform a motion correction, note this is just to make the
        # motion correction object with the correct displacment, but
        # the displacements should be zero here. These are manulally
        # added in the tetsts.
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

        return mc_recording_list, mc_motion_info_list  # , shifts

    ###########################################################################
    # Functional Tests
    ############################################################################

    @pytest.mark.parametrize("histogram_type", ["activity_1d", "activity_2d"])
    @pytest.mark.parametrize("num_shifts_global", [None, 200])
    def test_align_sessions_finds_correct_shifts(self, num_shifts_global, recording_1, histogram_type):
        """
        Test that `align_sessions` recovers the correct (linear) shifts.
        """
        recordings_list, shifts, peaks_list, peak_locations_list = recording_1

        assert shifts == (
            (0, 0),
            (0, -200),
            (0, 150),
        ), "expected shifts are hard-coded into this test ahould should be set in the fixture.."

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

            if DEBUG:
                from spikeinterface.widgets import plot_session_alignment, plot_activity_histogram_2d
                import matplotlib.pyplot as plt

                plot = plot_session_alignment(
                    recordings_list,
                    peaks_list,
                    peak_locations_list,
                    extra_info["session_histogram_list"],
                    **extra_info["corrected"],
                    spatial_bin_centers=extra_info["spatial_bin_centers"],
                    drift_raster_map_kwargs={"clim": (-250, 0), "scatter_decimate": 10},
                )
                plt.show()

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

    def test_histogram_generation(self, recording_1):
        """ """
        recordings_list, _, peaks_list, peak_locations_list = recording_1

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
    def test_histogram_parameters(self, recording_1, histogram_type, operator):
        """ """
        recordings_list, _, peaks_list, peak_locations_list = recording_1

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
    # Following Motion Correction
    ###########################################################################
    # These tests check that the displacement found by the inter-session alignment
    # is correctly added to any existing motion-correction results.

    def test_rigid_motion_rigid_intersession(self, recording_1):
        """
        Create an inter-session alignment recording and motion correct it so that
        it is an InterpolateMotion recording. Add some shifts to the existing displacement
        on the InterpolateMotion recordings and check the inter-session alignment shifts
        are properly added to this.
        """
        recordings_list, shifts, _, _ = recording_1

        mc_recording_list, mc_motion_info_list = self.motion_correct_recordings_list(
            recordings_list,
            rigid_motion=True,
        )
        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        # Ensure the motion was generated rigid by the test suite
        assert first_ses_mc_displacement[0].size == 1
        assert second_ses_mc_displacement[0].size == 1

        # Add some shifts to represent an existing motion correction
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

        # Check that the shift is the existing motion correction + the inter-session shift
        assert first_ses_total_displacement == [np.array([[shifts[0][1] + 0.01]])]
        assert second_ses_total_displacement == [np.array([[shifts[1][1] + 0.02]])]

        self.assert_interpolate_recording_not_duplicate(corrected_recordings[0])

    def test_rigid_motion_nonrigid_intersession(self, recording_2):
        """
        Test that non-rigid shifts estimated in inter-session alignment are properly
        added to rigid shifts estimated in motion correction.
        """
        recordings_list, _, peaks_list, _ = recording_2

        mc_recording_list, mc_motion_info_list = self.motion_correct_recordings_list(
            recordings_list,
            rigid_motion=True,
        )

        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        # Ensure the motion was generated rigid by the test suite
        assert first_ses_mc_displacement[0].size == 1
        assert second_ses_mc_displacement[0].size == 1

        # Add some shifts to represent an existing motion correction
        first_ses_mc_displacement[0] += 0.01
        second_ses_mc_displacement[0] += 0.02

        # All of this is direct copy between these tests...
        non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
        non_rigid_window_kwargs["rigid"] = False

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

        # The shift themselves are not expected to be correct to align this tricky test case
        # (see test_interesting_debug_case) but the shift on the motion objects should
        # match the estimateed shifts from inter-session alignment + the motion shifts set above
        assert np.all(extra_info["shifts_array"][0] + 0.01 == first_ses_total_displacement)
        assert np.all(extra_info["shifts_array"][1] + 0.02 == second_ses_total_displacement)

        self.assert_interpolate_recording_not_duplicate(corrected_recordings[0])

    @pytest.mark.parametrize("rigid_intersession", [True, False])
    def test_nonrigid_motion(self, rigid_intersession, recording_1, recording_2):
        """
        Now test that non-rigid motion estimates are properly combined with the
        rigid or non-rigid inter-session alignment estimates.
        """
        if rigid_intersession:
            recordings_list, _, peaks_list, _ = recording_1
        else:
            recordings_list, _, peaks_list, _ = recording_2

        mc_recording_list, mc_motion_info_list = self.motion_correct_recordings_list(
            recordings_list,
            rigid_motion=False,
        )

        # Now the motion data has multiple displcements (per probe segment window).
        # Add offsets to these different windows.
        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        offsets1 = np.linspace(0, 0.1, first_ses_mc_displacement[0].size)
        offsets2 = np.linspace(0, 0.1, first_ses_mc_displacement[0].size)

        first_ses_mc_displacement[0] += offsets1
        second_ses_mc_displacement[0] += offsets2

        # Now run inter-session alignment (either rigid or non-rigid). Check that
        # the final displacement on the motion objects is a combination of the
        # nonrigid motion estimate and rigid or nonrigid inter-session alignment estimate.
        non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
        non_rigid_window_kwargs["rigid"] = rigid_intersession

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

        assert np.all(extra_info["shifts_array"][0] + offsets1 == first_ses_total_displacement)
        assert np.all(extra_info["shifts_array"][1] + offsets2 == second_ses_total_displacement)

        self.assert_interpolate_recording_not_duplicate(corrected_recordings[0])

    def assert_interpolate_recording_not_duplicate(self, recording):
        """
        Do a quick check that indeed the interpolate recording is not duplicate
        (i.e. only one interpolate recording, and the previous is the generated simulation
        recording.
        """
        assert (
            isinstance(recording, InterpolateMotionRecording)
            and recording._parent.name == "InterSessionDisplacementRecording"
        )

    def test_motion_correction_peaks_are_converted(self, mocker, recording_1):
        """
        When `align_sessions_after_motion_correction` is run, the peaks locations
        used should be those that are already motion corrected, which requires
        correcting the peak locations in the function.

        Therefore, check that the final peak locations passed to `align_sessions`
        are motion-corrected.
        """
        recordings_list, _, peaks_list, peak_locations_list = recording_1

        # Motion correct recordings, and add a known motion-displacement
        mc_recording_list, mc_motion_info_list = self.motion_correct_recordings_list(
            recordings_list,
            rigid_motion=True,
        )

        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        first_ses_mc_displacement[0] += 0.1
        second_ses_mc_displacement[0] += 0.2

        # mock the `align_sessions` function to check what was passed
        spy_align_sessions = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.session_alignment, "align_sessions"
        )

        # Call the function, and check that the passed peak-locations are corrected
        corrected_recordings, _ = session_alignment.align_sessions_after_motion_correction(
            mc_recording_list, mc_motion_info_list, None
        )

        passed_peak_locations_1 = spy_align_sessions.call_args_list[0][0][2][0]
        passed_peak_locations_2 = spy_align_sessions.call_args_list[0][0][2][1]

        assert np.allclose(
            passed_peak_locations_1["y"], mc_motion_info_list[0]["peak_locations"]["y"] - 0.1, rtol=0, atol=1e-4
        )
        assert np.allclose(
            passed_peak_locations_2["y"], mc_motion_info_list[1]["peak_locations"]["y"] - 0.2, rtol=0, atol=1e-4
        )

    def test_motion_correction_kwargs(self, mocker, recording_1):
        """
        For `align_sessions_after_motion_correction`, if the motion-correct is non-rigid
        then the non-rigid window kwargs must match for inter-session alignment,
        otherwise it will not be possible to add the displacement.
        """
        recordings_list, _, _, _ = recording_1

        mc_recording_list, mc_motion_info_list = self.motion_correct_recordings_list(
            recordings_list,
            rigid_motion=False,
        )

        spy_align_sessions = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.session_alignment, "align_sessions"
        )

        # Run `align_sessions_after_motion_correction` with non-rigid window kwargs
        # that do not mach those used for motion correction
        changed = session_alignment.get_non_rigid_window_kwargs()
        changed["rigid"] = False
        changed["win_step_um"] = 51

        session_alignment.align_sessions_after_motion_correction(
            mc_recording_list, mc_motion_info_list, {"non_rigid_window_kwargs": changed}
        )

        # Now remove kwargs from the motion-correct and inter-session alignment (passed) non-rigid
        # window kwargs that don't match (Some from motion are not relevant for inter-session alignment,
        # some for inter-session and not set on motion (they may? predate their introduction).
        # Check that these core kwargs match (i.e. align_sessions is using the non-rigid-window
        # settings that motion uses.
        non_rigid_windows = mc_motion_info_list[0]["parameters"]["estimate_motion_kwargs"]
        non_rigid_windows.pop("method")
        non_rigid_windows.pop("bin_s")
        non_rigid_windows.pop(
            "hist_margin_um"
        )  # TODO: I think some kwargs are not exposed, this is probably okay, could mention to Sam

        passed_non_rigid_windows = spy_align_sessions.call_args_list[0][1]["non_rigid_window_kwargs"]
        passed_non_rigid_windows.pop("zero_threshold")
        passed_non_rigid_windows.pop("win_margin_um")

        assert sorted(passed_non_rigid_windows) == sorted(non_rigid_windows)

    ###########################################################################
    # Unit Tests
    ###########################################################################

    def test_shift_array_fill_zeros(self):
        """
        The tested function shifts a 1d array or 2d array (along a certain axis)
        and fills space with zero. Check that arrays are shifted as expected.
        """
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
        """
        Given a 'session matrix' of shifts (a matrix Mij where each element
        is the shift to get from session i to session j). It is skew-symmetric.
        """
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
    def test_compute_histogram_crosscorrelation_rigid(self, interpolate, odd_hist_size, shifts):
        """
        Create some toy array and shift it, then check that the cross-correlattion
        correctly finds the shifts, under a number of conditions.
        """
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
            min_crosscorr_threshold=0.001,
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

    @pytest.mark.parametrize("histogram_mode", ["activity_1d", "activity_2d"])
    def test_compute_histogram_crosscorrelation_nonrigid(self, histogram_mode):
        """ """
        # fmt: off
        #                 Window 1       | Window 2       | Window 3
        hist_1 = np.array([0.5, 1, 0, 0,   0, 1e-3, 0, 0,       0, 0, 0, 1])
        hist_2 = np.array([0, 0, 0.5, 1,   1e-12, 0, 0, 0,   1, 0, 0, 0])

        if histogram_mode == "activity_2d":
            hist_1 = np.vstack([hist_1, hist_1 * 2]).T
            hist_2 = np.vstack([hist_2, hist_2 * 2]).T

        winds = np.array([[ 1, 1, 1, 1,     0, 0, 0, 0,       0, 0, 0, 0],
                          [ 0, 0, 0, 0,     1, 1, 1, 1,       0, 0, 0, 0],
                          [ 0, 0, 0, 0,     0, 0, 0, 0,       1, 1, 1, 1]])

        shifts_matrix, xcorr_matrix_unsmoothed = alignment_utils.compute_histogram_crosscorrelation(
            np.stack([hist_1, hist_2], axis=0),
            non_rigid_windows=winds,
            num_shifts=None,
            interpolate=None,
            interp_factor=1.0,
            kriging_sigma=0.2,
            kriging_p=2,
            kriging_d=2,
            smoothing_sigma_bin=None,
            smoothing_sigma_window=None,
            min_crosscorr_threshold=0.001,
        )
        # fmt: on

        wind_1_shift = shifts_matrix[1, 0][0]
        wind_2_shift = shifts_matrix[1, 0][1]
        wind_3_shift = shifts_matrix[1, 0][2]

        assert wind_1_shift == 2  # the first window is shifted +2
        assert wind_2_shift == 0  # the second window has poor correlation under 0.001, so set to 0
        assert wind_3_shift == -3  # the third window is shifted -3

    ###########################################################################
    # Kwargs Tests
    ###########################################################################

    def test_get_estimate_histogram_kwargs(self, mocker, recording_1):

        recordings_list, _, peaks_list, peak_locations_list = recording_1

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

    def test_compute_alignment_kwargs(self, mocker, recording_1):

        recordings_list, _, peaks_list, peak_locations_list = recording_1

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
            "min_crosscorr_threshold": 0.001,
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

    def test_non_rigid_window_kwargs(self, mocker, recording_1):

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

        recordings_list, _, peaks_list, peak_locations_list = recording_1

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

    def test_interpolate_motion_kwargs(self, mocker, recording_1):
        """ """

        default_kwargs = {
            "border_mode": "force_zeros",
            "spatial_interpolation_method": "kriging",
            "sigma_um": 20.0,
            "p": 2,
        }
        assert (
            session_alignment.get_interpolate_motion_kwargs() == default_kwargs
        ), "Default `get_non_rigid_window_kwargs` were changed."

        different_kwargs = {
            "border_mode": "force_zeros",
            "spatial_interpolation_method": "nearest",
            "sigma_um": 25.0,
            "p": 3,
        }

        recordings_list, _, peaks_list, peak_locations_list = recording_1

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

    @pytest.mark.parametrize("histogram_type", ["activity_1d", "activity_2d"])
    def test_interesting_debug_case(self, histogram_type, recording_2):
        """
        This is an interseting debug case that is included in the tests to act as
        both a regression test and highlight how the alignment works and can lead
        to imperfect results in the test setting.

        In this case we take a non-rigid alignment, and we see that the right-edge
        of the histogram is aligned well, but the middle area (which lies within
        the same nonrigid window) is not well aligned. This is in spite of the
        shifts being estimated correctly for that segment.

        The problem is that the nonrigid bins are interpolated to get the shifts
        for each channel. In this case, the histogram peak being aligned
        lies in between two nonrigid window middle points, and so is interpolated.
        It is in the window with shift ~144 but next to a window with ~190
        and so ends up around ~170, resulting in the incorrect shift for this segment.
        But, in the real world it is necessary to interpolate channels like this
        or shifting whole windows would end up with very strange results!
        """
        recording_list, _, _, _ = recording_2

        mc_recording_list, mc_motion_info_list = self.motion_correct_recordings_list(recording_list, rigid_motion=True)

        # Run alignment
        non_rigid_window_kwargs = session_alignment.get_non_rigid_window_kwargs()
        non_rigid_window_kwargs["rigid"] = False
        non_rigid_window_kwargs["win_shape"] = "rect"
        non_rigid_window_kwargs["win_step_um"] = 250
        non_rigid_window_kwargs["win_scale_um"] = 250

        compute_alignment_kwargs = session_alignment.get_compute_alignment_kwargs()
        compute_alignment_kwargs["num_shifts_block"] = 75
        compute_alignment_kwargs["smoothing_sigma_bin"] = None
        compute_alignment_kwargs["smoothing_sigma_window"] = None

        estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
        estimate_histogram_kwargs["histogram_type"] = histogram_type

        corrected_recordings, extra_info = session_alignment.align_sessions_after_motion_correction(
            mc_recording_list,
            mc_motion_info_list,
            align_sessions_kwargs={
                "alignment_order": "to_session_1",  # to_center
                "non_rigid_window_kwargs": non_rigid_window_kwargs,
                "compute_alignment_kwargs": compute_alignment_kwargs,
                "estimate_histogram_kwargs": estimate_histogram_kwargs,
            },
        )

        if DEBUG:
            from spikeinterface.widgets import plot_session_alignment, plot_activity_histogram_2d
            import matplotlib.pyplot as plt

            # Plot the results, as well as the shift non-rigid window centers and
            # the shifts amount. You can see where the shift is reduced to align
            # the peaks in the middle of the histogram, the orange peak is
            # not sufficiently moved (because of the interpolation).
            peaks_list = [info["peaks"] for info in mc_motion_info_list]
            peak_locations_list = [info["peak_locations"] for info in mc_motion_info_list]
            plot = plot_session_alignment(
                mc_recording_list,
                peaks_list,
                peak_locations_list,
                extra_info["session_histogram_list"],
                **extra_info["corrected"],
                spatial_bin_centers=extra_info["spatial_bin_centers"],
                drift_raster_map_kwargs={"clim": (-250, 0), "scatter_decimate": 10},
            )

            window_edges = np.r_[0, np.cumsum(np.diff(extra_info["non_rigid_window_centers"]))]
            window_edges[:-1] += np.diff(window_edges) / 2

            y_window = extra_info["shifts_array"][1]
            x_bin = extra_info["non_rigid_window_centers"]

            ax4_twin = plot.figure.axes[4].twinx()
            ax5_twin = plot.figure.axes[5].twinx()
            ax4_twin.scatter(x_bin, y_window, color="red", s=100, edgecolor="black", zorder=3, label="Points")
            ax5_twin.scatter(x_bin, y_window, color="red", s=100, edgecolor="black", zorder=3, label="Points")

            for x_n, y_n in zip(x_bin, y_window):
                ax4_twin.plot([x_n, x_n], [0, y_n], color="black", linestyle="--", linewidth=1.5, zorder=2)
                ax5_twin.plot([x_n, x_n], [0, y_n], color="black", linestyle="--", linewidth=1.5, zorder=2)

            plt.suptitle("test_interesting_debug_case")
            plt.show()

        correced_hist_list = extra_info["corrected"]["corrected_session_histogram_list"]

        # This is a basic regression test to check this case does not change across versions.
        # Ideally this would be maintained, but it may be that these values need to change slightly.
        # In this case, the values can be changed or the regression part of this test removed, the
        # main value is in the debugging explaination.
        if histogram_type == "activity_1d":
            assert np.isclose(np.corrcoef(correced_hist_list[0], correced_hist_list[1])[0, 1], 0.283, atol=1e-3)
        else:
            assert np.isclose(
                np.corrcoef(np.mean(correced_hist_list[0], axis=1), np.mean(correced_hist_list[1], axis=1))[0, 1],
                0.372,
                atol=1e-3,
            )
