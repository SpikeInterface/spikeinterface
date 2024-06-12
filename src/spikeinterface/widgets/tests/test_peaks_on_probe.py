import pytest
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.widgets import plot_peaks_on_probe
from spikeinterface import generate_ground_truth_recording  # TODO: think about imports
import numpy as np


class TestPeaksOnProbe:

    @pytest.fixture(scope="session")
    def peak_info(self):
        """
        Fixture (created only once per test run) of a small
        ground truth recording with peaks and peak locations calculated.
        """
        recording, _ = generate_ground_truth_recording(num_units=5, num_channels=16, durations=[20, 9], seed=0)
        peaks = detect_peaks(recording)

        peak_locations = localize_peaks(
            recording,
            peaks,
            ms_before=0.3,
            ms_after=0.6,
            method="center_of_mass",
        )

        return (recording, peaks, peak_locations)

    def data_from_widget(self, widget, axes_idx):
        """
        Convenience function to get the data of the peaks
        that are on the plot (not sure why they are in the
        second 'collections').
        """
        return widget.axes[axes_idx].collections[2].get_offsets().data

    def test_peaks_on_probe_main(self, peak_info):
        """
        Plot all peaks, and check every peak is plot.
        Check the labels are corect.
        """
        recording, peaks, peak_locations = peak_info

        widget = plot_peaks_on_probe(recording, peaks, peak_locations, decimate=1)

        ax_y_data = self.data_from_widget(widget, 0)[:, 1]
        ax_y_pos = peak_locations["y"]

        assert np.array_equal(np.sort(ax_y_data), np.sort(ax_y_pos))
        assert widget.axes[0].get_ylabel() == "y ($\\mu m$)"
        assert widget.axes[0].get_xlabel() == "x ($\\mu m$)"

    @pytest.mark.parametrize("segment_index", [0, 1])
    def test_segment_selection(self, peak_info, segment_index):
        """
        Check that that when specifying only to plot peaks
        from a sepecific segment, that only peaks
        from that segment are plot.
        """
        recording, peaks, peak_locations = peak_info

        widget = plot_peaks_on_probe(
            recording,
            peaks,
            peak_locations,
            decimate=1,
            segment_index=segment_index,
        )

        ax_y_data = self.data_from_widget(widget, 0)[:, 1]
        ax_y_pos = peak_locations["y"][peaks["segment_index"] == segment_index]

        assert np.array_equal(np.sort(ax_y_data), np.sort(ax_y_pos))

    def test_multiple_inputs(self, peak_info):
        """
        Check that multiple inputs are correctly plot
        on separate axes. Do this my creating a copy
        of the peaks / peak locations with less peaks
        and different locations, for good measure.
        Check that these separate peaks / peak locations
        are plot on different axes.
        """
        recording, peaks, peak_locations = peak_info

        half_num_peaks = int(peaks.shape[0] / 2)

        peaks_change = peaks.copy()[:half_num_peaks]
        locs_change = peak_locations.copy()[:half_num_peaks]
        locs_change["y"] += 1

        widget = plot_peaks_on_probe(
            recording,
            [peaks, peaks_change],
            [peak_locations, locs_change],
            decimate=1,
        )

        # Test the first entry, axis 0
        ax_0_y_data = self.data_from_widget(widget, 0)[:, 1]

        assert np.array_equal(np.sort(peak_locations["y"]), np.sort(ax_0_y_data))

        # Test the second entry, axis 1.
        ax_1_y_data = self.data_from_widget(widget, 1)[:, 1]

        assert np.array_equal(np.sort(locs_change["y"]), np.sort(ax_1_y_data))

    def test_times_all(self, peak_info):
        """
        Check that when the times of peaks to plot is restricted,
        only peaks within the given time range are plot. Set the
        limits just before and after the second peak, and check only
        that peak is plot.
        """
        recording, peaks, peak_locations = peak_info

        peak_idx = 1
        peak_cutoff_low = peaks["sample_index"][peak_idx] - 1
        peak_cutoff_high = peaks["sample_index"][peak_idx] + 1

        widget = plot_peaks_on_probe(
            recording,
            peaks,
            peak_locations,
            decimate=1,
            time_range=(
                peak_cutoff_low / recording.get_sampling_frequency(),
                peak_cutoff_high / recording.get_sampling_frequency(),
            ),
        )

        ax_y_data = self.data_from_widget(widget, 0)[:, 1]

        assert np.array_equal([peak_locations[peak_idx]["y"]], ax_y_data)

    def test_times_per_segment(self, peak_info):
        """
        Test that the time bounds for multi-segment recordings
        with different times are handled properly. The time bounds
        given must respect the times for each segment. Here, we build
        two segments with times 0-100s and 100-200s. We set the
        time limits for peaks to plot as 50-150 i.e. all peaks
        from the second half of the first segment, and the first half
        of the second segment, should be plotted.

        Recompute peaks here for completeness even though this does
        duplicate the fixture.
        """
        recording, _, _ = peak_info

        first_seg_times = np.linspace(0, 100, recording.get_num_samples(0))
        second_seg_times = np.linspace(100, 200, recording.get_num_samples(1))

        recording.set_times(first_seg_times, segment_index=0)
        recording.set_times(second_seg_times, segment_index=1)

        # After setting the peak times above, re-detect peaks and plot
        # with a time range 50-150 s
        peaks = detect_peaks(recording)

        peak_locations = localize_peaks(
            recording,
            peaks,
            ms_before=0.3,
            ms_after=0.6,
            method="center_of_mass",
        )

        widget = plot_peaks_on_probe(
            recording,
            peaks,
            peak_locations,
            decimate=1,
            time_range=(
                50,
                150,
            ),
        )

        # Find the peaks that are expected to be plot given the time
        # restriction (second half of first segment, first half of
        # second segment) and check that indeed the expected locations
        # are displayed.
        seg_one_num_samples = recording.get_num_samples(0)
        seg_two_num_samples = recording.get_num_samples(1)

        okay_peaks_one = np.logical_and(
            peaks["segment_index"] == 0, peaks["sample_index"] > int(seg_one_num_samples / 2)
        )
        okay_peaks_two = np.logical_and(
            peaks["segment_index"] == 1, peaks["sample_index"] < int(seg_two_num_samples / 2)
        )
        okay_peaks = np.logical_or(okay_peaks_one, okay_peaks_two)

        ax_y_data = self.data_from_widget(widget, 0)[:, 1]

        assert any(okay_peaks), "someting went wrong in test generation, no peaks within the set time bounds detected"

        assert np.array_equal(np.sort(ax_y_data), np.sort(peak_locations[okay_peaks]["y"]))

    def test_get_min_and_max_times_in_recording(self, peak_info):
        """
        Check that the function which finds the minimum and maximum times
        across all segments in the recording returns correctly. First
        set times of the segments such that the earliest time is 50s and
        latest 200s. Check the function returns (50, 200).
        """
        recording, peaks, peak_locations = peak_info

        first_seg_times = np.linspace(50, 100, recording.get_num_samples(0))
        second_seg_times = np.linspace(100, 200, recording.get_num_samples(1))

        recording.set_times(first_seg_times, segment_index=0)
        recording.set_times(second_seg_times, segment_index=1)

        widget = plot_peaks_on_probe(
            recording,
            peaks,
            peak_locations,
            decimate=1,
        )

        min_max_times = widget._get_min_and_max_times_in_recording(recording)

        assert min_max_times == (50, 200)

    def test_ylim(self, peak_info):
        """
        Specify some y-axis limits (which is the probe height
        to show) and check that the plot is restricted to
        these limits.
        """
        recording, peaks, peak_locations = peak_info

        widget = plot_peaks_on_probe(
            recording,
            peaks,
            peak_locations,
            decimate=1,
            ylim=(300, 600),
        )

        assert widget.axes[0].get_ylim() == (300, 600)

    def test_decimate(self, peak_info):
        """
        By default, only a subset of peaks are shown for
        performance reasons. In tests, decimate is set to 1
        to ensure all peaks are plot. This tests now
        checks the decimate argument, to ensure peaks that are
        plot are correctly decimated.
        """
        recording, peaks, peak_locations = peak_info

        decimate = 5

        widget = plot_peaks_on_probe(
            recording,
            peaks,
            peak_locations,
            decimate=decimate,
        )

        ax_y_data = self.data_from_widget(widget, 0)[:, 1]
        ax_y_pos = peak_locations["y"][::decimate]

        assert np.array_equal(np.sort(ax_y_data), np.sort(ax_y_pos))

    def test_errors(self, peak_info):
        """
        Test all validation errors are raised when data in
        incorrect form is passed to the plotting function.
        """
        recording, peaks, peak_locations = peak_info

        # All lists must be same length
        with pytest.raises(ValueError) as e:
            plot_peaks_on_probe(
                recording,
                [peaks, peaks],
                [peak_locations],
            )

        # peaks and corresponding peak locations must be same size
        with pytest.raises(ValueError) as e:
            plot_peaks_on_probe(
                recording,
                [peaks[:-1]],
                [peak_locations],
            )

        # if one is list, both must be lists
        with pytest.raises(ValueError) as e:
            plot_peaks_on_probe(
                recording,
                peaks,
                [peak_locations],
            )

        # must have some peaks within the given time / segment
        with pytest.raises(ValueError) as e:
            plot_peaks_on_probe(recording, [peaks[:-1]], [peak_locations], time_range=(0, 0.001))
