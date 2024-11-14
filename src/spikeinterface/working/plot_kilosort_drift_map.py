from pathlib import Path
import matplotlib.axis
import scipy.signal

# from spikeinterface.core import read_python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats
import load_kilosort_utils

from spikeinterface.widgets.base import BaseWidget, to_attr


class KilosortDriftMapWidget(BaseWidget):
    """
    Create a drift map plot in the kilosort style. This is ported from Nick Steinmetz's
    `spikes` repository MATLAB code, https://github.com/cortex-lab/spikes.
    By default, a raster plot is drawn with the y-axis is spike depth and
    x-axis is time. Optionally, a corresponding 2D activity histogram can be
    added as a subplot (spatial bins, spike counts) with optional
    peak coloring and drift event detection (see below).
    Parameters
    ----------
    sorter_output : str | Path,
        Path to the kilosort output folder.
    only_include_large_amplitude_spikes : bool
        If `True`, only spikes with larger amplitudes are included. For
        details, see `_filter_large_amplitude_spikes()`.
    decimate : None | int
        If an integer n, only every nth spike is kept from the plot. Useful for improving
        performance when there are many spikes. If `None`, spikes will not be decimated.
    add_histogram_plot : bool
        If `True`, an activity histogram will be added to a new subplot to the
        left of the drift map.
    add_histogram_peaks_and_boundaries : bool
        If `True`, activity histogram peaks are detected and colored red if
        isolated according to start/end boundaries of the peak (blue otherwise).
    add_drift_events : bool
        If `True`, drift events will be plot on the raster map. Required
        `add_histogram_plot` and `add_histogram_peaks_and_boundaries` to run.
    weight_histogram_by_amplitude : bool
        If `True`, histogram counts will be weighted by spike amplitude.
    localised_spikes_only : bool
        If `True`, only spatially isolated spikes will be included.
    exclude_noise : bool
        If `True`, units labelled as noise in the `cluster_groups` file
        will be excluded.
    gain : float | None
        If not `None`, amplitudes will be scaled by the supplied gain.
    large_amplitude_only_segment_size: float
        If `only_include_large_amplitude_spikes` is `True`, the probe is split into
        segments to compute mean and std used as threshold. This sets the size of the
        segments in um.
    localised_spikes_channel_cutoff: int
        If `localised_spikes_only` is `True`, spikes that have more than half of the
        maximum loading channel over a range of > n channels are removed.
        This sets the number of channels.
    """

    def __init__(
        self,
        sorter_output: str | Path,
        only_include_large_amplitude_spikes: bool = True,
        decimate: None | int = None,
        add_histogram_plot: bool = False,
        add_histogram_peaks_and_boundaries: bool = True,
        add_drift_events: bool = True,
        weight_histogram_by_amplitude: bool = False,
        localised_spikes_only: bool = False,
        exclude_noise: bool = False,
        gain: float | None = None,
        large_amplitude_only_segment_size: float = 800.0,
        localised_spikes_channel_cutoff: int = 20,
    ):
        if not isinstance(sorter_output, Path):
            sorter_output = Path(sorter_output)

        if not sorter_output.is_dir():
            raise ValueError(f"No output folder found at {sorter_output}")

        if not (sorter_output / "params.py").is_file():
            raise ValueError(
                "The `sorting_output` path is not a valid kilosort output"
                "folder. It does not contain a `params.py` file`."
            )

        plot_data = dict(
            sorter_output=sorter_output,
            only_include_large_amplitude_spikes=only_include_large_amplitude_spikes,
            decimate=decimate,
            add_histogram_plot=add_histogram_plot,
            add_histogram_peaks_and_boundaries=add_histogram_peaks_and_boundaries,
            add_drift_events=add_drift_events,
            weight_histogram_by_amplitude=weight_histogram_by_amplitude,
            localised_spikes_only=localised_spikes_only,
            exclude_noise=exclude_noise,
            gain=gain,
            large_amplitude_only_segment_size=large_amplitude_only_segment_size,
            localised_spikes_channel_cutoff=localised_spikes_channel_cutoff,
        )
        BaseWidget.__init__(self, plot_data, backend="matplotlib")

    def plot_matplotlib(self, data_plot: dict, **unused_kwargs) -> None:

        dp = to_attr(data_plot)

        spike_indexes, spike_amplitudes, spike_locations, _ = load_kilosort_utils.compute_spike_amplitude_and_depth(
            dp.sorter_output, dp.localised_spikes_only, dp.exclude_noise, dp.gain, dp.localised_spikes_channel_cutoff
        )
        spike_times = spike_indexes / 30000
        spike_depths = spike_locations[:, 1]

        # Calculate the amplitude range for plotting first, so the scale is always the
        # same across all options (e.g. decimation) which helps with interpretability.
        if dp.only_include_large_amplitude_spikes:
            amplitude_range_all_spikes = (
                spike_amplitudes.min(),
                spike_amplitudes.max(),
            )
        else:
            amplitude_range_all_spikes = np.percentile(spike_amplitudes, (1, 90))

        if dp.decimate:
            spike_times = spike_times[:: dp.decimate]
            spike_amplitudes = spike_amplitudes[:: dp.decimate]
            spike_depths = spike_depths[:: dp.decimate]

        if dp.only_include_large_amplitude_spikes:
            spike_times, spike_amplitudes, spike_depths = self._filter_large_amplitude_spikes(
                spike_times, spike_amplitudes, spike_depths, dp.large_amplitude_only_segment_size
            )

        # Setup axis and plot the raster drift map
        fig = plt.figure(figsize=(10, 10 * (6 / 8)))

        if dp.add_histogram_plot:
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])
            hist_axis = fig.add_subplot(gs[0])
            raster_axis = fig.add_subplot(gs[1], sharey=hist_axis)
        else:
            raster_axis = fig.add_subplot()

        self._plot_kilosort_drift_map_raster(
            spike_times,
            spike_amplitudes,
            spike_depths,
            amplitude_range_all_spikes,
            axis=raster_axis,
        )

        if not dp.add_histogram_plot:
            raster_axis.set_xlabel("time")
            raster_axis.set_ylabel("y position")
            self.axes = [raster_axis]
            return

        # If the histogram plot is requested, plot it alongside
        # it's peak colouring, bounds display and drift point display.
        hist_axis.set_xlabel("count")
        raster_axis.set_xlabel("time")
        hist_axis.set_ylabel("y position")

        bin_centers, counts = self._compute_activity_histogram(
            spike_amplitudes, spike_depths, dp.weight_histogram_by_amplitude
        )
        hist_axis.plot(counts, bin_centers, color="black", linewidth=1)

        if dp.add_histogram_peaks_and_boundaries:
            drift_events = self._color_histogram_peaks_and_detect_drift_events(
                spike_times, spike_depths, counts, bin_centers, hist_axis
            )

            if dp.add_drift_events and np.any(drift_events):
                raster_axis.scatter(drift_events[:, 0], drift_events[:, 1], facecolors="r", edgecolors="none")
                for i, _ in enumerate(drift_events):
                    raster_axis.text(
                        drift_events[i, 0] + 1, drift_events[i, 1], str(np.round(drift_events[i, 2])), color="r"
                    )
        self.axes = [hist_axis, raster_axis]

    def _plot_kilosort_drift_map_raster(
        self,
        spike_times: np.ndarray,
        spike_amplitudes: np.ndarray,
        spike_depths: np.ndarray,
        amplitude_range: np.ndarray | tuple,
        axis: matplotlib.axes.Axes,
    ) -> None:
        """
        Plot a drift raster plot in the kilosort style.
        This function was ported from Nick Steinmetz's `spikes` repository
        MATLAB code, https://github.com/cortex-lab/spikes
        Parameters
        ----------
        spike_times : np.ndarray
            (num_spikes,) array of spike times.
        spike_amplitudes : np.ndarray
                (num_spikes,) array of corresponding spike amplitudes.
        spike_depths : np.ndarray
                (num_spikes,) array of corresponding spike depths.
        amplitude_range : np.ndarray | tuple
            (2,) array of min, max amplitude values for color binning.
        axis : matplotlib.axes.Axes
            Matplotlib axes object on which to plot the drift map.
        """
        n_color_bins = 20
        marker_size = 0.5

        color_bins = np.linspace(amplitude_range[0], amplitude_range[1], n_color_bins)

        colors = plt.get_cmap("gray")(np.linspace(0, 1, n_color_bins))[::-1]

        for bin_idx in range(n_color_bins - 1):

            spikes_in_amplitude_bin = np.logical_and(
                spike_amplitudes >= color_bins[bin_idx], spike_amplitudes <= color_bins[bin_idx + 1]
            )
            axis.scatter(
                spike_times[spikes_in_amplitude_bin],
                spike_depths[spikes_in_amplitude_bin],
                color=colors[bin_idx],
                s=marker_size,
                antialiased=True,
            )

    def _compute_activity_histogram(
        self, spike_amplitudes: np.ndarray, spike_depths: np.ndarray, weight_histogram_by_amplitude: bool
    ) -> tuple[np.ndarray, ...]:
        """
        Compute the activity histogram for the kilosort drift map's left-side plot.
        Parameters
        ----------
        spike_amplitudes : np.ndarray
            (num_spikes,) array of spike amplitudes.
        spike_depths : np.ndarray
            (num_spikes,) array of spike depths.
        weight_histogram_by_amplitude : bool
            If `True`, the spike amplitudes are taken into consideration when generating the
            histogram. The amplitudes are scaled to the range [0, 1] then summed for each bin,
            to generate the histogram values. If `False`, counts (i.e. num spikes per bin)
            are used.
        Returns
        -------
        bin_centers : np.ndarray
            The spatial bin centers (probe depth) for the histogram.
        values : np.ndarray
            The histogram values. If `weight_histogram_by_amplitude` is `False`, these
            values represent are counts, otherwise they are counts weighted by amplitude.
        """
        assert (
            spike_amplitudes.dtype == np.float64
        ), "`spike amplitudes should be high precision as many values are summed."

        bin_um = 2
        bins = np.arange(spike_depths.min() - bin_um, spike_depths.max() + bin_um, bin_um)
        values, bins = np.histogram(spike_depths, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if weight_histogram_by_amplitude:
            bin_indices = np.digitize(spike_depths, bins, right=True) - 1
            values = np.zeros(bin_indices.max() + 1, dtype=np.float64)
            scaled_spike_amplitudes = (spike_amplitudes - spike_amplitudes.min()) / np.ptp(spike_amplitudes)
            np.add.at(values, bin_indices, scaled_spike_amplitudes)

        return bin_centers, values

    def _color_histogram_peaks_and_detect_drift_events(
        self,
        spike_times: np.ndarray,
        spike_depths: np.ndarray,
        counts: np.ndarray,
        bin_centers: np.ndarray,
        hist_axis: matplotlib.axes.Axes,
    ) -> np.ndarray:
        """
        Given an activity histogram, color the peaks red (isolated peak) or
        blue (peak overlaps with other peaks) and compute spatial drift
        events for isolated peaks across time bins.
        This function was ported from Nick Steinmetz's `spikes` repository
        MATLAB code, https://github.com/cortex-lab/spikes
        Parameters
        ----------
        spike_times : np.ndarray
            (num_spikes,) array of spike times.
        spike_depths : np.ndarray
            (num_spikes,) array of corresponding spike depths.
        counts : np.ndarray
            (num_bins,) array of histogram bin counts.
        bin_centers : np.ndarray
            (num_bins,) array of histogram bin centers.
        hist_axis : matplotlib.axes.Axes
            Axes on which the histogram is plot, to add peaks.
        Returns
        -------
        drift_events : np.ndarray
            A (num_drift_events, 3) array of drift events. The columns are
            (time_position, spatial_position, drift_value). The drift
            value is computed per time, spatial bin as the difference between
            the median position of spikes in the bin, and the bin center.
        """
        all_peak_indexes = scipy.signal.find_peaks(
            counts,
        )[0]

        # Filter low-frequency peaks, so they are not included in the
        # step to determine whether peaks are overlapping (new step
        # introduced in the port to python)
        bin_above_freq_threshold = counts[all_peak_indexes] > 0.3 * spike_times[-1]
        filtered_peak_indexes = all_peak_indexes[bin_above_freq_threshold]

        drift_events = []
        for idx, peak_index in enumerate(filtered_peak_indexes):

            peak_count = counts[peak_index]

            # Find the start and end of peak min/max bounds (5% of amplitude)
            start_position = np.where(counts[:peak_index] < peak_count * 0.05)[0].max()
            end_position = np.where(counts[peak_index:] < peak_count * 0.05)[0].min() + peak_index

            if (  # bounds include another, different histogram peak
                idx > 0
                and start_position < filtered_peak_indexes[idx - 1]
                or idx < filtered_peak_indexes.size - 1
                and end_position > filtered_peak_indexes[idx + 1]
            ):
                hist_axis.scatter(peak_count, bin_centers[peak_index], facecolors="none", edgecolors="blue")
                continue

            else:
                for position in [start_position, end_position]:
                    hist_axis.axhline(bin_centers[position], 0, counts.max(), color="grey", linestyle="--")
                hist_axis.scatter(peak_count, bin_centers[peak_index], facecolors="none", edgecolors="red")

                # For isolated histogram peaks, detect the drift events, defined as
                # difference between spatial bin center and median spike depth in the bin
                # over 6 um (in time / spatial bins with at least 10 spikes).
                depth_in_window = np.logical_and(
                    spike_depths > bin_centers[start_position],
                    spike_depths < bin_centers[end_position],
                )
                current_spike_depths = spike_depths[depth_in_window]
                current_spike_times = spike_times[depth_in_window]

                window_s = 10

                all_time_bins = np.arange(0, np.ceil(spike_times[-1]).astype(int), window_s)
                for time_bin in all_time_bins:

                    spike_in_time_bin = np.logical_and(
                        current_spike_times >= time_bin, current_spike_times <= time_bin + window_s
                    )
                    drift_size = bin_centers[peak_index] - np.median(current_spike_depths[spike_in_time_bin])

                    # 6 um is the hardcoded threshold for drift, and we want at least 10 spikes for the median calculation
                    bin_has_drift = np.abs(drift_size) > 6 and np.sum(spike_in_time_bin, dtype=np.int16) > 10
                    if bin_has_drift:
                        drift_events.append((time_bin + window_s / 2, bin_centers[peak_index], drift_size))

        drift_events = np.array(drift_events)

        return drift_events

    def _filter_large_amplitude_spikes(
        self,
        spike_times: np.ndarray,
        spike_amplitudes: np.ndarray,
        spike_depths: np.ndarray,
        large_amplitude_only_segment_size,
    ) -> tuple[np.ndarray, ...]:
        """
        Return spike properties with only the largest-amplitude spikes included. The probe
        is split into egments, and within each segment the mean and std computed.
        Any spike less than 1.5x the standard deviation in amplitude of it's segment is excluded
        Splitting the probe is only done for the exclusion step, the returned array are flat.
        Takes as input arrays `spike_times`, `spike_depths` and `spike_amplitudes` and returns
        copies of these arrays containing only the large amplitude spikes.
        """
        spike_bool = np.zeros_like(spike_amplitudes, dtype=bool)

        segment_size_um = large_amplitude_only_segment_size

        probe_segments_left_edges = np.arange(np.floor(spike_depths.max() / segment_size_um) + 1) * segment_size_um

        for segment_left_edge in probe_segments_left_edges:
            segment_right_edge = segment_left_edge + segment_size_um

            spikes_in_seg = np.where(
                np.logical_and(spike_depths >= segment_left_edge, spike_depths < segment_right_edge)
            )[0]
            spike_amps_in_seg = spike_amplitudes[spikes_in_seg]
            is_high_amplitude = spike_amps_in_seg > np.mean(spike_amps_in_seg) + 1.5 * np.std(spike_amps_in_seg, ddof=1)

            spike_bool[spikes_in_seg] = is_high_amplitude

        spike_times = spike_times[spike_bool]
        spike_amplitudes = spike_amplitudes[spike_bool]
        spike_depths = spike_depths[spike_bool]

        return spike_times, spike_amplitudes, spike_depths


KilosortDriftMapWidget(
    "/Users/joeziminski/data/bombcelll/sorter_output",
    only_include_large_amplitude_spikes=False,
    localised_spikes_only=True,
)
plt.show()

"""
        sorter_output: str | Path,
        only_include_large_amplitude_spikes: bool = True,
        decimate: None | int = None,
        add_histogram_plot: bool = False,
        add_histogram_peaks_and_boundaries: bool = True,
        add_drift_events: bool = True,
        weight_histogram_by_amplitude: bool = False,
        localised_spikes_only: bool = False,
        exclude_noise: bool = False,
        gain: float | None = None,
        large_amplitude_only_segment_size: float = 800.0,
        localised_spikes_channel_cutoff: int = 20,
"""
