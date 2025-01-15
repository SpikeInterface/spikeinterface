import numpy as np
import pytest

from spikeinterface.preprocessing.inter_session_alignment import session_alignment
from spikeinterface.generation.session_displacement_generator import *


class TestInterSessionAlignment:
    """ """

    @pytest.fixture(scope="session")
    def test_recording_1(self):
        """ """
        shifts = ((0, 0), (0, -200), (0, 150))

        recordings_list, _ = generate_session_displacement_recordings(
            num_units=15,
            recording_durations=[9, 10, 11],
            recording_shifts=shifts,
            # TODO: can see how well this is recaptured by comparing the displacements to the known displacement + gradient
            non_rigid_gradient=None,  # 0.1, # 0.1,
            seed=55,  # 52
            generate_sorting_kwargs=dict(firing_rates=(100, 250), refractory_period_ms=4.0),
            generate_unit_locations_kwargs=dict(
                margin_um=0.0,
                # if this is say 20, then units go off the edge of the probe and are such low amplitude they are not picked up.
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

    # TEST 1D AND 2D HERE
    def test_align_sessions_finds_correct_shifts(self, test_recording_1):
        """ """
        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        # try num units 5 and 65

        compute_alignment_kwargs = session_alignment.get_compute_alignment_kwargs()
        compute_alignment_kwargs["smoothing_sigma_bin"] = None
        compute_alignment_kwargs["smoothing_sigma_window"] = None

        estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
        estimate_histogram_kwargs["bin_um"] = 0.5

        for mode, expected in zip(
            ["to_session_1", "to_session_2", "to_session_3", "to_middle"],
            [
                (0, -200, 150),
                (200, 0, 350),
                (-150, -350, 0),
                (16.66, -183.33, 166.66),
            ],  # TODO: hard coded from shifts...
        ):
            corrected_recordings_list, extra_info = session_alignment.align_sessions(
                recordings_list,
                peaks_list,
                peak_locations_list,
                alignment_order=mode,
                compute_alignment_kwargs=compute_alignment_kwargs,
                estimate_histogram_kwargs=estimate_histogram_kwargs,
            )
            assert np.allclose(expected, extra_info["shifts_array"].squeeze(), rtol=0, atol=1.5)

        corr_peaks_list, corr_peak_loc_list = session_alignment.compute_peaks_locations_for_session_alignment(
            corrected_recordings_list,
            detect_kwargs={"method": "locally_exclusive"},
            localize_peaks_kwargs={"method": "grid_convolution"},
        )

        import matplotlib.pyplot as plt

        new_histograms = session_alignment._compute_session_histograms(
            corrected_recordings_list, corr_peaks_list, corr_peak_loc_list, **estimate_histogram_kwargs
        )[0]

        try:
            assert np.all(
                np.abs(np.corrcoef(new_histograms)) - np.abs(np.corrcoef(extra_info["session_histogram_list"])) >= 0
            )  ## TODO: just compare upper triangular...
        except:
            breakpoint()

        if False:  # the problem here is that border_mode is zero and now this seems to
            import os  # lead to some cases

            os.environ["hello_world"] = ""

            _, extra_info = align_sessions(
                corrected_recordings_list,
                corr_peaks_list,
                corr_peak_loc_list,
                alignment_order="to_session_1",
                compute_alignment_kwargs=compute_alignment_kwargs,
                estimate_histogram_kwargs=estimate_histogram_kwargs,
            )
            plot_session_alignment(
                corrected_recordings_list,
                corr_peaks_list,
                corr_peak_loc_list,
                extra_info["session_histogram_list"],
                **extra_info["corrected"],
                spatial_bin_centers=extra_info["spatial_bin_centers"],  # TODO: why have to pass this..?
                drift_raster_map_kwargs={"clim": (-250, 0), "scatter_decimate": 10},
            )
            plt.show()

    # _compute_session_histograms
    # ##################################################################################

    # test histogram is accurate (1D and 2D) (TEST MEAN VS. MEDIAN FOR 1D AND 2D
    # test chunked histogram is accurate (1D and 2D)
    # test all arguments are passed down to the correct subfunction
    # test spatial bin centers
    # unit test get_activity_histogram extra arguments.

    def test_histogram_generation(self, test_recording_1):
        """ """
        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        recording = recordings_list[0]

        channel_locations = recording.get_channel_locations()
        loc_start = np.min(channel_locations[:, 1])
        loc_end = np.max(channel_locations[:, 1])

        # TODO: test the case where unit locations "y" are outside of the probe!

        chunk_time_window = 1  # parameterize
        bin_um = 1  # parameterize

        estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
        estimate_histogram_kwargs["bin_um"] = bin_um
        estimate_histogram_kwargs["chunked_bin_size_s"] = chunk_time_window  # TODO: maybe in separate?

        (
            session_histogram_list,
            temporal_bin_centers_list,
            spatial_bin_centers,  # X
            spatial_bin_edges,  # X
            histogram_info_list,
        ) = session_alignment._compute_session_histograms(
            recordings_list, peaks_list, peak_locations_list, **estimate_histogram_kwargs
        )

        num_bins = (loc_end - loc_start) / bin_um  ## TODO: what to do when this is not integer!? TEST CASE!

        np.min(peak_locations_list[0]["y"])
        np.max(peak_locations_list[0]["y"])  # TODO: compare against loc_end...

        peak_locations = peak_locations_list[0]

        bin_edges = np.linspace(loc_start, loc_end, int(num_bins) + 1)  # TODO: test non-integer case
        bin_centers = bin_edges[:-1] + bin_um / 2

        assert np.array_equal(bin_edges, spatial_bin_edges)
        assert np.array_equal(bin_centers, spatial_bin_centers)

        for recording, temporal_bin_center in zip(recordings_list, temporal_bin_centers_list):
            times = recording.get_times()
            centers = (np.max(times) - np.min(times)) / 2
            assert temporal_bin_center == centers

        # dict_keys(['chunked_histograms', 'chunked_temporal_bin_centers', 'chunked_bin_size_s'])

        for ses_idx, (recording, chunked_histogram_info) in enumerate(zip(recordings_list, histogram_info_list)):

            # TODO: this is direct copy from above, can merge
            times = recording.get_times()

            num_windows = (np.round(np.max(times)) - np.min(times)) / chunk_time_window  # round or ceil?
            temp_bin_edges = np.linspace(np.min(times), np.round(np.max(times)), int(num_windows) + 1)
            centers = temp_bin_edges[:-1] + bin_um / 2

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

                assert np.array_equal(
                    np.histogram(new_peak_locs["y"], bins=bin_edges)[0] / (upper - lower),
                    chunked_histogram_info["chunked_histograms"][edge_idx, :],
                )

        for ses_idx in range(len(session_histogram_list)):
            assert np.array_equal(
                session_histogram_list[ses_idx],
                np.histogram(peak_locations_list[ses_idx]["y"], bins=bin_edges)[0]
                / recordings_list[ses_idx].get_duration(),
            )

        # TODO: check median vs mean (simply compute on the chunked_histograms, which are tested here)

    # TODO:
    # test the main histogram (variable bin num)
    # test the chunked histogram (split the peak locs based on the sample idx!)
    # find peaks list within bin_edges
    # scale to Hz :)

    # 1) test all sub-functions are called with the correct arguments:
    # do this by getter

    # concat data and see if it imporves sorting. this is more benchmarking...
    # test histogram generation from known peaks, peak locations
    # test temporal histogram bin size (chunked), in general histogram_info
    # but test main histogram time window too

    # test after motion correction (all 4 possibilities) that shifts are correctly added, inspect the motion object correctly.

    # test shifts_array are correct (known value) and that corrected peak locations list, corrected session histogram list are okay.
    # test for 'to session X' and 'to middle'
    # ^^ only need to test rigid only

    # test nonrigid bin sizes are correct (rect vs. gaussian) and non_rigid_windows and window_centers respect the passed arguments
    # somehow check the histogram or histogram list? really need to check the output...

    # Somehow test nonrigid is better... (maybe create one with rigid and
    # see a small improvement)

    # test spatial bin centeres, with known probe, and different bin sizes

    ###########################################################################
    # Kwargs Tests
    ###########################################################################

    # TODO: requires pytest-mock
    def test_get_estimate_histogram_kwargs(self, mocker, test_recording_1):
        #  breakpoint()

        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        default_kwargs = {
            "bin_um": 2,
            "method": "chunked_mean",
            "chunked_bin_size_s": "estimate",
            "log_scale": False,
            "depth_smooth_um": None,
            "histogram_type": "activity_1d",
            "weight_with_amplitude": False,
            "avg_in_bin": False,  # TODO
        }

        assert default_kwargs == session_alignment.get_estimate_histogram_kwargs()

        different_kwargs = {  #      Lowest Function Call
            "bin_um": 5,  # make_2d_motion_histogram
            "method": "chunked_median",  # _get_single_session_activity_histogram
            "chunked_bin_size_s": 6,  # make_2d_motion_histogram (bin_s)
            "log_scale": True,  # get_activity_histogram
            "depth_smooth_um": 5,  # make_2d_motion_histogram
            "histogram_type": "activity_1d",  # _get_single_session_activity_histogram
            "weight_with_amplitude": True,  # make_2d_motion_histogram
            "avg_in_bin": True,  #              # make_2d_motion_histogram
        }

        import spikeinterface  # require for monkeypatch, import here to avoid confusion

        spy_2d_histogram = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.alignment_utils, "make_2d_motion_histogram"
        )
        spy_get_activity_histogram = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.alignment_utils, "get_activity_histogram"
        )
        spy_compute_histogram = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.session_alignment,
            "_get_single_session_activity_histogram",
        )

        session_alignment.align_sessions(
            [recordings_list[0]], [peaks_list[0]], [peak_locations_list[0]], estimate_histogram_kwargs=different_kwargs
        )
        # get_activity_histogram for log_scale
        first_call = spy_2d_histogram.call_args_list[0]
        args, kwargs = first_call

        assert kwargs["bin_s"] == different_kwargs["chunked_bin_size_s"]
        assert kwargs["bin_um"] is None
        assert np.unique(np.diff(kwargs["spatial_bin_edges"])) == different_kwargs["bin_um"]
        assert kwargs["depth_smooth_um"] == different_kwargs["depth_smooth_um"]
        assert kwargs["weight_with_amplitude"] == different_kwargs["weight_with_amplitude"]
        assert kwargs["avg_in_bin"] == different_kwargs["avg_in_bin"]

        # TODO: these might be overkill, these are implicitly tested elsewhere...
        # Keep for now but maybe replace, and only monkeypatch calls that
        # in which the functions are not tested here! Yes obvs that is much better...
        first_call = spy_get_activity_histogram.call_args_list[0]
        args, kwargs = first_call
        assert kwargs["log_scale"] == different_kwargs["log_scale"]

        first_call = spy_compute_histogram.call_args_list[0]
        args, kwargs = first_call
        assert kwargs["method"] == different_kwargs["method"]
        assert kwargs["histogram_type"] == different_kwargs["histogram_type"]

    def test_compute_alignment_kwargs(self, mocker, test_recording_1):

        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        import spikeinterface  # TODO: move

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
        assert session_alignment.get_compute_alignment_kwargs() == default_kwargs

        different_kwargs = {
            "num_shifts_global": 1,
            "num_shifts_block": 5,
            "interpolate": True,
            "interp_factor": 15,
            "kriging_sigma": 20,
            "kriging_p": 21,
            "kriging_d": 22,
            "smoothing_sigma_bin": 1.2,
            "smoothing_sigma_window": 1.3,
            "akima_interp_nonrigid": True,  # TOOD: test elsewhere
        }

        spy_compute_alignment_kwargs = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.alignment_utils, "compute_histogram_crosscorrelation"
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

        kwargs = spy_compute_alignment_kwargs.call_args_list[0][1]

        assert kwargs["num_shifts"] == different_kwargs["num_shifts_global"]
        assert kwargs["interpolate"] == different_kwargs["interpolate"]
        assert kwargs["interp_factor"] == different_kwargs["interp_factor"]
        assert kwargs["kriging_sigma"] == different_kwargs["kriging_sigma"]
        assert kwargs["kriging_p"] == different_kwargs["kriging_p"]
        assert kwargs["kriging_d"] == different_kwargs["kriging_d"]
        assert kwargs["smoothing_sigma_bin"] == different_kwargs["smoothing_sigma_bin"]
        assert kwargs["smoothing_sigma_window"] == different_kwargs["smoothing_sigma_window"]

        kwargs = spy_compute_alignment_kwargs.call_args_list[1][1]
        assert kwargs["num_shifts"] == different_kwargs["num_shifts_block"]

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
        assert session_alignment.get_non_rigid_window_kwargs() == default_kwargs

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
        assert session_alignment.get_interpolate_motion_kwargs() == default_kwargs

        different_kwargs = {
            "border_mode": "force_zeros",  # fixed as this until can figure out probe
            "spatial_interpolation_method": "nearest",
            "sigma_um": 25.0,
            "p": 3,
        }

        import spikeinterface  # MOVE TO TOP

        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        spy_get_activity_histogram = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.session_alignment.InterpolateMotionRecording,
            "__init__",
        )
        session_alignment.align_sessions(
            [recordings_list[0]], [peaks_list[0]], [peak_locations_list[0]], interpolate_motion_kwargs=different_kwargs
        )
        first_call = spy_get_activity_histogram.call_args_list[0]
        args, kwargs = first_call

        assert kwargs["border_mode"] == different_kwargs["border_mode"]
        assert kwargs["spatial_interpolation_method"] == different_kwargs["spatial_interpolation_method"]
        assert kwargs["sigma_um"] == different_kwargs["sigma_um"]
        assert kwargs["p"] == different_kwargs["p"]

    ###########################################################################
    # Following Motion Correction
    ###########################################################################

    def get_motion_corrected_recordings_list(self, rigid_motion=True, rigid_intersession=True):
        # TODO: explicitly ensure data are not multi-segment
        shifts = ((0, 0), (0, 250))

        non_rigid_gradient = None if rigid_intersession else 0.2
        recordings_list, _ = generate_session_displacement_recordings(
            num_units=5,
            recording_durations=[1, 1],
            recording_shifts=shifts,
            # TODO: can see how well this is recaptured by comparing the displacements to the known displacement + gradient
            non_rigid_gradient=non_rigid_gradient,  # 0.1, # 0.1,
            seed=55,  # 52
            generate_sorting_kwargs=dict(firing_rates=(100, 250), refractory_period_ms=4.0),
            generate_unit_locations_kwargs=dict(
                margin_um=0.0,
                # if this is say 20, then units go off the edge of the probe and are such low amplitude they are not picked up.
                minimum_z=0.0,
                maximum_z=2.0,
                minimum_distance=18.0,
                max_iteration=100,
                distance_strict=False,
            ),
            generate_noise_kwargs=dict(noise_levels=(0.0, 0.0), spatial_decay=1.0),
        )

        interpolate_motion_kwargs = {"border_mode": "force_zeros"}

        preset = "rigid_fast" if rigid_motion else "kilosort_like"

        import spikeinterface.full as si

        # mc_recording_list = [si.correct_motion(rec, preset=preset, interpolate_motion_kwargs=interpolate_motion_kwargs) for rec in recordings_list]
        mc_recording_list = []
        mc_motion_info_list = []
        for rec in recordings_list:
            corrected_rec, motion_info = si.correct_motion(
                rec, preset=preset, interpolate_motion_kwargs=interpolate_motion_kwargs, output_motion_info=True
            )
            mc_recording_list.append(corrected_rec)
            mc_motion_info_list.append(motion_info)

        # its okay that these displacements are zero?

        peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
            mc_recording_list,
            detect_kwargs={"method": "locally_exclusive"},
            localize_peaks_kwargs={"method": "grid_convolution"},  # TODO: faster methods
        )

        return mc_recording_list, mc_motion_info_list, peaks_list, peak_locations_list, shifts

    def test_interpolate_motion_parameters(self):
        pass

    def test_rigid_motion_rigid_intersession(self, test_recording_1):
        """ """
        mc_recording_list, mc_motion_info_list, peaks_list, peak_locations_list, shifts = (
            self.get_motion_corrected_recordings_list(rigid_motion=True, rigid_intersession=False)
        )

        # should assert these are zero!
        first_ses_mc_displacement = mc_recording_list[0]._recording_segments[0].motion.displacement
        second_ses_mc_displacement = mc_recording_list[1]._recording_segments[0].motion.displacement

        assert first_ses_mc_displacement[0].size == 1
        assert second_ses_mc_displacement[0].size == 1

        #      first_ses_mc_displacement[0] += 0.01
        #     second_ses_mc_displacement[0] += 0.02

        corrected_recordings, extra_info = session_alignment.align_sessions_after_motion_correction(
            mc_recording_list,
            mc_motion_info_list,
            align_sessions_kwargs={"alignment_order": "to_session_1"},
        )

        # TODO: peaks_list does not match mc_motion_info_list peaks!!!
        breakpoint()

        first_ses_total_displacement = (
            corrected_recordings[0]._recording_segments[0].motion.displacement
        )  # [array([[-124.9, -124.9, -124.9, -124.9, -124.9]])]
        second_ses_total_displacement = corrected_recordings[1]._recording_segments[0].motion.displacement

        breakpoint()  # TODO: something has gone wrong!!
        assert first_ses_total_displacement == [np.array([[0.01]])]  # TODO: can use shifts directly!
        assert second_ses_total_displacement == [np.array([[250.02]])]

    def test_rigid_motion_nonrigid_intersession(self):

        # TODO: DIRECT COPY!
        mc_recording_list, mc_motion_info_list, peaks_list, peak_locations_list, shifts = (
            self.get_motion_corrected_recordings_list(rigid_motion=True, rigid_intersession=False)
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

        first_ses_total_displacement = (
            corrected_recordings[0]._recording_segments[0].motion.displacement
        )  # [array([[-124.9, -124.9, -124.9, -124.9, -124.9]])]
        second_ses_total_displacement = corrected_recordings[1]._recording_segments[0].motion.displacement

        assert np.all(extra_info["shifts_array"][0] + 0.01 == first_ses_total_displacement)
        assert np.all(extra_info["shifts_array"][1] + 0.02 == second_ses_total_displacement)  #  [np.array([[250.02]])]

    # TODO: Does this really test the passed peak list, peak_locations list? maybe use a monkeypatch?
    @pytest.mark.parametrize("rigid_intersession", [True, False])
    def test_nonrigid_motion(self, rigid_intersession):
        # TODO: DIRECT COPY!
        mc_recording_list, mc_motion_info_list, peaks_list, peak_locations_list, shifts = (
            self.get_motion_corrected_recordings_list(rigid_motion=False, rigid_intersession=rigid_intersession)
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

        first_ses_total_displacement = (
            corrected_recordings[0]._recording_segments[0].motion.displacement
        )  # [array([[-124.9, -124.9, -124.9, -124.9, -124.9]])]
        second_ses_total_displacement = corrected_recordings[1]._recording_segments[0].motion.displacement

        breakpoint()

        assert np.all(extra_info["shifts_array"][0] + offsets1 == first_ses_total_displacement)
        assert np.all(
            extra_info["shifts_array"][1] + offsets2 == second_ses_total_displacement
        )  # [np.array([[250.02]])]

    def test_nonrigid_motion_nonrigid_intersession(self):
        pass

    ###########################################################################
    # Unit Tests
    ###########################################################################

    ###########################################################################
    # Benchmarking
    ###########################################################################

    # 1) create some inter-session drift
    # 2) compare sorting before / after
