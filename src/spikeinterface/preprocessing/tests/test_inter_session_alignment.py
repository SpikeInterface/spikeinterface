import numpy as np
import pytest

from spikeinterface.preprocessing.inter_session_alignment import session_alignment
from spikeinterface.generation.session_displacement_generator import *

class TestInterSessionAlignment:
    """

    """

    @pytest.fixture(scope="session")
    def test_recording_1(self):
        """
        """
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
            generate_noise_kwargs=dict(noise_levels=(0.0, 0.0), spatial_decay=1.0)
        )

        peaks_list, peak_locations_list = session_alignment.compute_peaks_locations_for_session_alignment(
            recordings_list, detect_kwargs={"method": "locally_exclusive"}, localize_peaks_kwargs={"method": "grid_convolution"},
        )
        return (recordings_list, shifts, peaks_list, peak_locations_list)

    ###########################################################################
    # Functional Tests
    ############################################################################

    # TEST 1D AND 2D HERE
    def test_align_sessions_finds_correct_shifts(self, test_recording_1):
        """
        """
        recordings_list, _,  peaks_list, peak_locations_list = test_recording_1

        # try num units 5 and 65

        compute_alignment_kwargs = session_alignment.get_compute_alignment_kwargs()
        compute_alignment_kwargs["smoothing_sigma_bin"] = None
        compute_alignment_kwargs["smoothing_sigma_window"] = None

        estimate_histogram_kwargs = session_alignment.get_estimate_histogram_kwargs()
        estimate_histogram_kwargs["bin_um"] = 0.5

        for mode, expected in zip(
            ["to_session_1", "to_session_2", "to_session_3", "to_middle"],
            [(0, -200, 150), (200, 0, 350), (-150, -350, 0), (16.66, -183.33, 166.66)]  # TODO: hard coded from shifts...
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
            corrected_recordings_list, detect_kwargs={"method": "locally_exclusive"}, localize_peaks_kwargs={"method": "grid_convolution"},
        )

        import matplotlib.pyplot as plt

        new_histograms = session_alignment._compute_session_histograms(
            corrected_recordings_list, corr_peaks_list, corr_peak_loc_list, **estimate_histogram_kwargs
        )[0]

        try:
            assert np.all(np.abs(np.corrcoef(new_histograms)) - np.abs(np.corrcoef(extra_info["session_histogram_list"])) >= 0)  ## TODO: just compare upper triangular...
        except:
            breakpoint()

        if False:      # the problem here is that border_mode is zero and now this seems to
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
                drift_raster_map_kwargs={"clim": (-250, 0), "scatter_decimate": 10}
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
        """
        """
        recordings_list, _, peaks_list, peak_locations_list = test_recording_1

        recording = recordings_list[0]

        channel_locations = recording.get_channel_locations()
        loc_start = np.min(channel_locations[:, 1])
        loc_end = np.max(channel_locations[:, 1])

        # TODO: test the case where unit locations "y" are outside of the probe!

        chunk_time_window = 1  # parameterize
        bin_um = 1             # parameterize

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
            temp_bin_edges = np.linspace(np.min(times), np.round(np.max(times)) , int(num_windows) + 1)
            centers = temp_bin_edges[:-1] + bin_um / 2

            assert chunked_histogram_info["chunked_bin_size_s"] == chunk_time_window
            assert np.array_equal(chunked_histogram_info["chunked_temporal_bin_centers"], centers)

            for edge_idx in range(len(temp_bin_edges) - 1):

                lower = temp_bin_edges[edge_idx]
                upper = temp_bin_edges[edge_idx + 1]

                lower_idx = recording.time_to_sample_index(lower)
                upper_idx = recording.time_to_sample_index(upper)

                new_peak_locs = peak_locations_list[ses_idx][np.where(np.logical_and(peaks_list[ses_idx]["sample_index"] >= lower_idx, peaks_list[ses_idx]["sample_index"] < upper_idx))  ]

                assert np.array_equal(np.histogram(new_peak_locs["y"], bins=bin_edges)[0] / (upper-lower), chunked_histogram_info["chunked_histograms"][edge_idx, : ])

        for ses_idx in range(len(session_histogram_list)):
            assert np.array_equal(session_histogram_list[ses_idx], np.histogram(peak_locations_list[ses_idx]["y"], bins=bin_edges)[0] / recordings_list[ses_idx].get_duration())

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

    def test_get_interpolate_motion_kwargs(self):
        pass

    def test_get_compute_alignment_kwargs(self):
        pass

    def test_get_non_rigid_window_kwargs(self):
        pass

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

        different_kwargs = {
            "bin_um": 5,
            "method": "chunked_median",
            "chunked_bin_size_s": 6,
            "log_scale": True,
            "depth_smooth_um": 5,
            "histogram_type": "activity_1d",
            "weight_with_amplitude": True,
            "avg_in_bin": True,  # TODO
        }

        from spikeinterface.sortingcomponents.motion.motion_utils import make_2d_motion_histogram
        # monkeypatch XXX
        import spikeinterface

        spy = mocker.spy(
            spikeinterface.preprocessing.inter_session_alignment.alignment_utils,
            'make_2d_motion_histogram'
        )
        session_alignment.align_sessions(
            [recordings_list[0]], [peaks_list[0]], [peak_locations_list[0]],
            estimate_histogram_kwargs=different_kwargs
        )

        assert spy.call_count == 2

        first_call = spy.call_args_list[0]
        args, kwargs = first_call

        assert kwargs["bin_s"] == different_kwargs["bin_s"]
        assert kwargs["bin_um"] == different_kwargs["bin_um"]
        assert kwargs["depth_smooth_um"] == different_kwargs["depth_smooth_um"]
        assert kwargs["weight_with_amplitude"] == different_kwargs["weight_with_amplitude"]
        assert kwargs["avg_in_bin"] == different_kwargs["avg_in_bin"]

        breakpoint()

        second_call = spy.call_args_list[1]
        args, kwargs = second_call
        breakpoint()

        # Check the arguments for each call
        for call in spy.call_args_list:
            args, kwargs = call

            breakpoint()
            # Add specific assertions for the expected arguments
            assert kwargs["bin_s"] == 6
            assert kwargs["bin_um"] == 5
            # Check other arguments as needed


        # Verify that make_2d_motion_histogram was called with the expected arguments
        spy.assert_called_once_with(
            bin_s=6,
            bin_um=5,
#            method="chunked_median",
  #          chunked_bin_size_s="5",
 #           log_scale=True,
            depth_smooth_um=5,
   #         histogram_type="activity_2d",
            weight_with_amplitude=True,
            avg_in_bin=True,  # Add any other arguments you expect here

            direction="y",
            time_smooth_s=None,
        )






    ###########################################################################
    # Unit Tests
    ###########################################################################
