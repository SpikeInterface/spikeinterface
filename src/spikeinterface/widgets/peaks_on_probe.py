from __future__ import annotations

import numpy as np


from .base import BaseWidget, to_attr


class PeaksOnProbeWidget(BaseWidget):
    """
    Generate a plot of spike peaks showing their location on a plot
    of the probe. Color scaling represents spike amplitude.

    The generated plot overlays the estimated position of a spike peak
    (as a single point for each peak) onto a plot of the probe. The
    dimensions of the plot are x axis: probe width, y axis: probe depth.

    Plots of different sets of peaks can be created on subplots, by
    passing a list of peaks and corresponding peak locations.

    Parameters
    ----------
    recording : Recording
        A SpikeInterface recording object.
    peaks : np.array | list[np.ndarray]
        SpikeInterface 'peaks' array created with `detect_peaks()`,
        an array of length num_peaks with entries:
            (sample_index, channel_index, amplitude, segment_index)
        To plot different sets of peaks in subplots, pass a list of peaks, each
        with a corresponding entry in a list passed to `peak_locations`.
    peak_locations : np.array | list[np.ndarray]
        A SpikeInterface 'peak_locations' array created with `localize_peaks()`.
        an array of length num_peaks with entries: (x, y)
        To plot multiple peaks in subplots, pass a list of `peak_locations`
        here with each entry having a corresponding `peaks`.
    segment_index : None | int, default: None
        If set, only peaks from this recording segment will be used.
    time_range : None | Tuple, default: None
        The time period over which to include peaks. If `None`, peaks
        across the entire recording will be shown.
    ylim : None | Tuple, default: None
        The y-axis limits (i.e. the probe depth). If `None`, the entire
        probe will be displayed.
    decimate : int, default: 5
        For performance reasons, every nth peak is shown on the plot,
        where n is set by decimate. To plot all peaks, set `decimate=1`.
    """

    def __init__(
        self,
        recording,
        peaks,
        peak_locations,
        segment_index=None,
        time_range=None,
        ylim=None,
        decimate=5,
        backend=None,
        **backend_kwargs,
    ):
        data_plot = dict(
            recording=recording,
            peaks=peaks,
            peak_locations=peak_locations,
            segment_index=segment_index,
            time_range=time_range,
            ylim=ylim,
            decimate=decimate,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from spikeinterface.widgets import plot_probe_map

        dp = to_attr(data_plot)

        peaks, peak_locations = self._check_and_format_inputs(
            dp.peaks,
            dp.peak_locations,
        )
        fs = dp.recording.get_sampling_frequency()
        num_plots = len(peaks)

        # Set the maximum time to the end time of the longest segment
        if dp.time_range is None:

            time_range = self._get_min_and_max_times_in_recording(dp.recording)
        else:
            time_range = dp.time_range

        ## Create the figure and axes
        if backend_kwargs["figsize"] is None:
            backend_kwargs.update(dict(figsize=(12, 8)))

        self.figure, self.axes, self.ax = make_mpl_figure(num_axes=num_plots, **backend_kwargs)
        self.axes = self.axes[0]

        # Plot each passed peaks / peak_locations over the probe on a separate subplot
        for ax_idx, (peaks_to_plot, peak_locs_to_plot) in enumerate(zip(peaks, peak_locations)):

            ax = self.axes[ax_idx]
            plot_probe_map(dp.recording, ax=ax)

            time_mask = self._get_peaks_time_mask(dp.recording, time_range, peaks_to_plot)

            if dp.segment_index is not None:
                segment_mask = peaks_to_plot["segment_index"] == dp.segment_index
                mask = time_mask & segment_mask
            else:
                mask = time_mask

            if not any(mask):
                raise ValueError(
                    "No peaks within the time and segment mask found. Change `time_range` or `segment_index`"
                )

            # only plot every nth peak
            peak_slice = slice(None, None, dp.decimate)

            # Find the amplitudes for the colormap scaling
            # (intensity represents amplitude)
            amps = np.abs(peaks_to_plot["amplitude"][mask][peak_slice])
            amps /= np.quantile(amps, 0.95)
            cmap = plt.get_cmap("inferno")(amps)
            color_kwargs = dict(alpha=0.2, s=2, c=cmap)

            # Plot the peaks over the plot, and set the y-axis limits.
            ax.scatter(
                peak_locs_to_plot["x"][mask][peak_slice], peak_locs_to_plot["y"][mask][peak_slice], **color_kwargs
            )

            if dp.ylim is None:
                padding = 25  # arbitary padding just to give some space around highests and lowest peaks on the plot
                ylim = (np.min(peak_locs_to_plot["y"]) - padding, np.max(peak_locs_to_plot["y"]) + padding)
            else:
                ylim = dp.ylim

            ax.set_ylim(ylim[0], ylim[1])

        self.figure.suptitle(f"Peaks on Probe Plot")

    def _get_peaks_time_mask(self, recording, time_range, peaks_to_plot):
        """
        Return a mask of `True` where the peak is within the given time range
        and `False` otherwise.

        This is a little complex, as each segment can have different start /
        end times. For each segment, find the time bounds relative to that
        segment time and fill the `time_mask` one segment at a time.
        """
        time_mask = np.zeros(peaks_to_plot.size, dtype=bool)

        for seg_idx in range(recording.get_num_segments()):

            segment = recording.select_segments(seg_idx)

            t_start_sample = segment.time_to_sample_index(time_range[0])
            t_stop_sample = segment.time_to_sample_index(time_range[1])

            seg_mask = peaks_to_plot["segment_index"] == seg_idx

            time_mask[seg_mask] = (t_start_sample < peaks_to_plot[seg_mask]["sample_index"]) & (
                peaks_to_plot[seg_mask]["sample_index"] < t_stop_sample
            )

        return time_mask

    def _get_min_and_max_times_in_recording(self, recording):
        """
        Find the maximum and minimum time across all segments in the recording.
        For example if the segment times are (10-100 s, 0 - 50s) the
        min and max times are (0, 100)
        """
        t_starts = []
        t_stops = []
        for seg_idx in range(recording.get_num_segments()):

            segment = recording.select_segments(seg_idx)

            t_starts.append(segment.sample_index_to_time(0))

            t_stops.append(segment.sample_index_to_time(segment.get_num_samples() - 1))

        time_range = (np.min(t_starts), np.max(t_stops))

        return time_range

    def _check_and_format_inputs(self, peaks, peak_locations):
        """
        Check that the inpust are in expected form. Corresponding peaks
        and peak_locations of same size and format must be provided.
        """
        types_are_list = [isinstance(peaks, list), isinstance(peak_locations, list)]

        if not all(types_are_list):
            if any(types_are_list):
                raise ValueError("`peaks` and `peak_locations` must either be both lists or both not lists.")
            peaks = [peaks]
            peak_locations = [peak_locations]

        if len(peaks) != len(peak_locations):
            raise ValueError(
                "If `peaks` and `peak_locations` are lists, they must contain "
                "the same number of (corresponding) peaks and peak locations."
            )

        for idx, (peak, peak_loc) in enumerate(zip(peaks, peak_locations)):
            if peak.size != peak_loc.size:
                raise ValueError(
                    f"The number of peaks and peak_locations do not "
                    f"match for the {idx} input. For each spike peak, there "
                    f"must be a corresponding peak location"
                )

        return peaks, peak_locations
