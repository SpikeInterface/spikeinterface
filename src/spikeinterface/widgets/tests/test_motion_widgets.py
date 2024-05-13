import copy

import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_preprocess
import spikeinterface.widgets as si_widgets
import matplotlib.pyplot as plt
import pytest
import numpy as np
import time


class TestMotionWidgets:
    """
    Tests the functionality and parameters of `plot_motion`
    function.

    Currently there are not tests covering
    that the corrected peak depth is correct, as
    in the `plot_motion` function the (tested) peak depth
    is passed directly to the correction function.

    Also, `amplitude_clim` is not tested.

    These tests use a `motion_data` fixture that runs
    only once per tests ("class scope"), as well as
    a fixture that closes the widget figure after
    each test to avoid a strange tkinkter bug that
    was appearing unpredictably.
    """

    @pytest.fixture(scope="class")
    def motion_data(self, tmp_path_factory):
        """
        Fixture to create a test recording and run
        motion correction (nonrigid).
        """
        rec, _ = si_extractors.toy_example(
            num_segments=1,
            duration=30.0,
            num_units=10,
            num_channels=12,
        )

        folder = tmp_path_factory.mktemp("test_motion_widgets")

        rec_corrected = si_preprocess.correct_motion(rec, folder=folder)
        motion_info = si_preprocess.load_motion_info(folder)

        motion_data = {"rec": rec, "rec_corrected": rec_corrected, "motion_info": motion_info}

        return motion_data

    @pytest.fixture(scope="function")
    def close_figures(self):
        """
        Closing figures after each test, and including a short pause,
        is required to avoid tkinter "init.tcl" access error that
        can randomly occur.
        """
        yield
        plt.close("all")

    # Check plots are shown as expected
    # -------------------------------------------------------------------------

    def test_plot_motion(self, motion_data, close_figures):
        """
        Test the expected axis (along with title and x/y labels)
        are created for nonrigid motion correction.
        """
        widget = si_widgets.plot_motion(
            motion_data["motion_info"],
            motion_data["rec_corrected"],
        )

        self.check_all_plot_labels(widget)

    def test_plot_motion_rigid(self, motion_data, tmp_path, close_figures):
        """
        Text the expected axes as above but for rigid motion correction.
        """
        rec_corrected = si_preprocess.correct_motion(motion_data["rec"], preset="rigid_fast", folder=tmp_path)

        motion_info = si_preprocess.load_motion_info(tmp_path)

        widget = si_widgets.plot_motion(
            motion_info,
            rec_corrected,
        )
        self.check_all_plot_labels(widget, rigid=True)

    def check_all_plot_labels(self, widget, rigid=False):
        """
        Helper function to check the expected axes along
        with title and x/y labels are contained
        a created widget.
        """
        first_plot = widget.axes[0]
        assert first_plot.get_title() == "Peak depth"
        assert first_plot.get_xlabel() == "Time [s]"
        assert first_plot.get_ylabel() == "Depth [μm]"

        second_plot = widget.axes[1]
        assert second_plot.get_title() == "Corrected peak depth"
        assert second_plot.get_xlabel() == "Time [s]"
        assert second_plot.get_ylabel() == "Depth [μm]"

        third_plot = widget.axes[2]
        assert third_plot.get_title() == "Motion vectors"
        assert third_plot.get_ylabel() == "Motion [μm]"
        assert third_plot.get_xlabel() == "Time [s]"

        if rigid:
            assert len(widget.axes) == 3
        else:
            fourth_plot = widget.axes[3]
            assert fourth_plot.get_title() == "Motion vectors"
            assert fourth_plot.get_ylabel() == "Depth [μm]"
            assert fourth_plot.get_xlabel() == "Time [s]"

    def test_peak_depth(self, motion_data, close_figures):
        """
        Check that the plot peak depth matches the
        actual peak locations of the `motion_info`.
        """
        widget = si_widgets.plot_motion(
            motion_data["motion_info"],
            motion_data["rec_corrected"],
        )

        assert np.array_equal(
            motion_data["motion_info"]["peak_locations"]["y"],
            widget.axes[0].collections[0].get_offsets().data[:, 1],
        ), "Peak depth does not match motion_info locations."

    def test_motion(self, motion_data, close_figures):
        """
        Check that the motion plots reflect the actual
        motion in `motion_info` (tested first channel only).
        """
        widget = si_widgets.plot_motion(
            motion_data["motion_info"],
            motion_data["rec_corrected"],
        )
        assert np.array_equal(
            motion_data["motion_info"]["motion"][:, 0], widget.axes[2].lines[0].get_ydata()
        ), "motion of first channel does not match plot."

    # Check plots are shown as expected
    # -------------------------------------------------------------------------

    def test_axis_limit_arguments(self, motion_data, close_figures):
        """
        Test the axis limits change as expected when set.
        """
        widget = si_widgets.plot_motion(
            motion_data["motion_info"],
            motion_data["rec_corrected"],
            depth_lim=(-200, 200),
            motion_lim=1.5,
        )
        for i in range(2):
            assert widget.axes[i].get_ylim() == (-200.0, 200.0), f"Plot {i} failed depth lim."

        assert widget.axes[2].get_ylim() == (-1.5, 1.5), "Failed motion lim."

    def test_decimate(self, motion_data, close_figures):
        """
        Test data is displayed as expected on the
        scatter plot if downsampled.
        """
        widget = si_widgets.plot_motion(
            motion_data["motion_info"],
            motion_data["rec_corrected"],
            scatter_decimate=10,
        )
        assert np.array_equal(
            motion_data["motion_info"]["peak_locations"]["y"][::10],
            widget.axes[0].collections[0].get_offsets().data[:, 1],
        ), "Peak depth does not match motion_info locations."

    def test_color_amplitude(self, motion_data, close_figures):
        """
        Test the color amplitude argument. This will color the
        scatter plot depending on the amplitude of the AP.

        This is a little tricky to test as the cmap is not 1:1
        based on amplitude but binned. Therefore, it is tested
        just to ensure that no amplitude in a color bin are
        larger than in the preceeding (lighter) bin.
        """
        decimate = 2
        alpha = 0.8

        widget = si_widgets.plot_motion(
            motion_data["motion_info"],
            motion_data["rec_corrected"],
            color_amplitude=True,
            scatter_decimate=decimate,
            amplitude_alpha=alpha,
            amplitude_cmap="grey",
        )

        amplitudes = motion_data["motion_info"]["peaks"]["amplitude"][::decimate]

        # Get the color bin and color bin id for each AP. Iterate through
        # the bins and check the intensity increaes with amplitude as
        # expected.
        uniques, color_bin_id = np.unique(widget.axes[0].collections[0].get_facecolors()[:, 0], return_inverse=True)

        for i in range(1, uniques.size):

            amp_this_level = amplitudes[np.where(color_bin_id == i)]
            amp_prev_level = amplitudes[np.where(color_bin_id == i - 1)]

            assert (np.abs(amp_prev_level).min() < np.abs(amp_this_level)).all()

        # Check alpha
        assert (widget.axes[0].collections[0].get_facecolors()[:, 3 == alpha]).all()

    def test_times(self, motion_data, close_figures):
        """
        Time can be set based on the recording or
        just assuming the time starts at zero. Test that time
        is displayed properly when the recording time does not
        start at zero.
        """
        orig_start = 0
        orig_end = motion_data["rec_corrected"].get_times()[-1]

        new_start = 10
        new_end = 20

        new_times = np.linspace(new_start, new_end, motion_data["rec_corrected"].get_num_samples())
        rec = copy.deepcopy(motion_data["rec_corrected"])
        rec.set_times(new_times)

        widget_with_times = si_widgets.plot_motion(motion_data["motion_info"], rec)

        widget_no_times = si_widgets.plot_motion(
            motion_data["motion_info"],
        )

        for i in range(2):
            test_new_start, test_new_end = widget_with_times.axes[i].get_xlim()

            assert np.isclose(test_new_start, new_start, rtol=0, atol=1)
            assert np.isclose(test_new_end, new_end, rtol=0, atol=1)

            test_orig_start, test_orig_end = widget_no_times.axes[2].get_xlim()
            assert not np.isclose(test_new_start, test_orig_start, rtol=0, atol=1)

            assert np.isclose(test_orig_start, orig_start, rtol=0, atol=1)
            assert np.isclose(test_orig_end, orig_end, rtol=0, atol=1)
