from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr

from spikeinterface.core import BaseRecording, SortingAnalyzer
from .rasters import BaseRasterWidget
from spikeinterface.core.motion import Motion


class MotionWidget(BaseWidget):
    """
    Plot the Motion object.

    Parameters
    ----------
    motion : Motion
        The motion object.
    segment_index : int | None, default: None
        If Motion is multi segment, the must be not None.
    mode : "auto" | "line" | "map", default: "line"
        How to plot the motion.
        "line" plots estimated motion at different depths as lines.
        "map" plots estimated motion at different depths as a heatmap.
        "auto" makes it automatic depending on the number of motion depths.
    """

    def __init__(
        self,
        motion: Motion,
        segment_index: int | None = None,
        mode: str = "line",
        motion_lim: float | None = None,
        backend: str | None = None,
        **backend_kwargs,
    ):
        if isinstance(motion, dict):
            raise ValueError(
                "The API has changed, plot_motion() used Motion object now, maybe you want plot_motion_info(motion_info)"
            )

        if segment_index is None:
            if len(motion.displacement) == 1:
                segment_index = 0
            else:
                raise ValueError("plot motion : the Motion object is multi segment you must provide segment_index=XX")

        plot_data = dict(
            motion=motion,
            segment_index=segment_index,
            mode=mode,
            motion_lim=motion_lim,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        assert backend_kwargs["axes"] is None

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        motion = dp.motion
        displacement = motion.displacement[dp.segment_index]
        temporal_bins_s = motion.temporal_bins_s[dp.segment_index]
        depth = motion.spatial_bins_um

        if dp.motion_lim is None:
            motion_lim = np.max(np.abs(displacement)) * 1.05
        else:
            motion_lim = dp.motion_lim

        ax = self.ax
        fig = self.figure
        if dp.mode == "line":
            ax.plot(temporal_bins_s, displacement, alpha=0.2, color="black")
            ax.plot(temporal_bins_s, np.mean(displacement, axis=1), color="C0")
            ax.set_xlabel("Times [s]")
            ax.set_ylabel("motion [um]")
        elif dp.mode == "map":
            im = ax.imshow(
                displacement.T,
                interpolation="nearest",
                aspect="auto",
                origin="lower",
                extent=(temporal_bins_s[0], temporal_bins_s[-1], depth[0], depth[-1]),
                cmap="PiYG",
            )
            im.set_clim(-motion_lim, motion_lim)

            cbar = fig.colorbar(im)
            cbar.ax.set_ylabel("motion [um]")
            ax.set_xlabel("Times [s]")
            ax.set_ylabel("Depth [um]")


class DriftRasterMapWidget(BaseRasterWidget):
    """
    Plot the drift raster map from peaks or a SortingAnalyzer.
    The drift raster map is a scatter plot of the estimated peak depth vs time and it is
    useful to visualize the drift over the course of the recording.

    Parameters
    ----------
    peaks : np.array | None, default: None
        The peaks array, with dtype ("sample_index", "channel_index", "amplitude", "segment_index"),
        as returned by the `detect_peaks` or `correct_motion` functions.
    peak_locations : np.array | None, default: None
        The peak locations, with dtype ("x", "y") or ("x", "y", "z"), as returned by the
        `localize_peaks` or `correct_motion` functions.
    sorting_analyzer : SortingAnalyzer | None, default: None
        The sorting analyzer object. To use this function, the `SortingAnalyzer` must have the
        "spike_locations" extension computed.
    direction : "x" or "y", default: "y"
        The direction to display. "y" is the depth direction.
    segment_index : int, default: None
        The segment index to display.
    recording : RecordingExtractor | None, default: None
        The recording extractor object (only used to get "real" times).
    segment_index : int, default: 0
        The segment index to display.
    sampling_frequency : float, default: None
        The sampling frequency (needed if recording is None).
    depth_lim : tuple or None, default: None
        The min and max depth to display, if None (min and max of the recording).
    scatter_decimate : int, default: None
        If equal to n, each nth spike is kept for plotting.
    color_amplitude : bool, default: True
        If True, the color of the scatter points is the amplitude of the peaks.
    cmap : str, default: "inferno"
        The colormap to use for the amplitude.
    color : str, default: "Gray"
        The color of the scatter points if color_amplitude is False.
    clim : tuple or None, default: None
        The min and max amplitude to display, if None (min and max of the amplitudes).
    alpha : float, default: 1
        The alpha of the scatter points.
    """

    def __init__(
        self,
        peaks: np.array | None = None,
        peak_locations: np.array | None = None,
        sorting_analyzer: SortingAnalyzer | None = None,
        direction: str = "y",
        recording: BaseRecording | None = None,
        sampling_frequency: float | None = None,
        segment_index: int | None = None,
        depth_lim: tuple[float, float] | None = None,
        color_amplitude: bool = True,
        scatter_decimate: int | None = None,
        cmap: str = "inferno",
        color: str = "Gray",
        clim: tuple[float, float] | None = None,
        alpha: float = 1,
        backend: str | None = None,
        **backend_kwargs,
    ):
        assert peaks is not None or sorting_analyzer is not None
        if peaks is not None:
            assert peak_locations is not None
            if recording is None:
                assert sampling_frequency is not None, "If recording is None, you must provide the sampling frequency"
            else:
                sampling_frequency = recording.sampling_frequency
            peak_amplitudes = peaks["amplitude"]
        if sorting_analyzer is not None:
            if sorting_analyzer.has_recording():
                recording = sorting_analyzer.recording
                sampling_frequency = recording.sampling_frequency
            else:
                recording = None
                sampling_frequency = sorting_analyzer.sampling_frequency
            peaks = sorting_analyzer.sorting.to_spike_vector()
            assert sorting_analyzer.has_extension(
                "spike_locations"
            ), "The sorting analyzer must have the 'spike_locations' extension to use this function"
            peak_locations = sorting_analyzer.get_extension("spike_locations").get_data()
            if color_amplitude:
                assert sorting_analyzer.has_extension("spike_amplitudes"), (
                    "The sorting analyzer must have the 'spike_amplitudes' extension to use color_amplitude=True. "
                    "You can compute it or set color_amplitude=False."
                )
            if sorting_analyzer.has_extension("spike_amplitudes"):
                peak_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()
            else:
                peak_amplitudes = None

        if segment_index is None:
            assert (
                len(np.unique(peaks["segment_index"])) == 1
            ), "segment_index must be specified if there are multiple segments"
            segment_index = 0
        else:
            peak_mask = peaks["segment_index"] == segment_index
            peaks = peaks[peak_mask]
            peak_locations = peak_locations[peak_mask]
            if peak_amplitudes is not None:
                peak_amplitudes = peak_amplitudes[peak_mask]

        from matplotlib.pyplot import colormaps

        if color_amplitude:
            amps = peak_amplitudes
            amps_abs = np.abs(amps)
            q_95 = np.quantile(amps_abs, 0.95)
            cmap = colormaps[cmap]
            if clim is None:
                amps = amps_abs
                amps /= q_95
                c = cmap(amps)
            else:
                from matplotlib.colors import Normalize

                norm_function = Normalize(vmin=clim[0], vmax=clim[1], clip=True)
                c = cmap(norm_function(amps))
            color_kwargs = dict(
                color=None,
                c=c,
                alpha=alpha,
            )
        else:
            color_kwargs = dict(color=color, c=None, alpha=alpha)

        # convert data into format that `BaseRasterWidget` can take it in
        spike_train_data = {0: peaks["sample_index"] / sampling_frequency}
        y_axis_data = {0: peak_locations[direction]}

        plot_data = dict(
            spike_train_data=spike_train_data,
            y_axis_data=y_axis_data,
            y_lim=depth_lim,
            color_kwargs=color_kwargs,
            scatter_decimate=scatter_decimate,
            title="Peak depth",
            y_label="Depth [um]",
        )

        BaseRasterWidget.__init__(self, **plot_data, backend=backend, **backend_kwargs)


class MotionInfoWidget(BaseWidget):
    """
    Plot motion information from the motion_info dictionary returned by the `correct_motion()` funciton.
    This widget plots:
        * the motion iself
        * the drift raster map (peak depth vs time) before correction
        * the drift raster map (peak depth vs time) after correction

    Parameters
    ----------
    motion_info : dict
        The motion info returned by correct_motion() or loaded back with load_motion_info().
    recording : RecordingExtractor
        The recording extractor object
    segment_index : int, default: None
        The segment index to display.
    sampling_frequency : float, default: None
        The sampling frequency (needed if recording is None).
    depth_lim : tuple or None, default: None
        The min and max depth to display, if None (min and max of the recording).
    motion_lim : tuple or None, default: None
        The min and max motion to display, if None (min and max of the motion).
    scatter_decimate : int, default: None
        If equal to n, each nth spike is kept for plotting.
    color_amplitude : bool, default: False
        If True, the color of the scatter points is the amplitude of the peaks.
    amplitude_cmap : str, default: "inferno"
        The colormap to use for the amplitude.
    amplitude_color : str, default: "Gray"
        The color of the scatter points if color_amplitude is False.
    amplitude_clim : tuple or None, default: None
        The min and max amplitude to display, if None (min and max of the amplitudes).
    amplitude_alpha : float, default: 1
        The alpha of the scatter points.
    """

    def __init__(
        self,
        motion_info: dict,
        recording: BaseRecording,
        segment_index: int | None = None,
        depth_lim: tuple[float, float] | None = None,
        motion_lim: tuple[float, float] | None = None,
        color_amplitude: bool = False,
        scatter_decimate: int | None = None,
        amplitude_cmap: str = "inferno",
        amplitude_color: str = "Gray",
        amplitude_clim: tuple[float, float] | None = None,
        amplitude_alpha: float = 1,
        backend: str | None = None,
        **backend_kwargs,
    ):

        motion = motion_info["motion"]
        if segment_index is None:
            if len(motion.displacement) == 1:
                segment_index = 0
            else:
                raise ValueError(
                    "plot drift map : the Motion object is multi-segment you must provide segment_index=XX"
                )
        assert recording.get_num_segments() == len(
            motion.displacement
        ), "The number of segments in the recording must be the same as the number of segments in the motion object"

        plot_data = dict(
            sampling_frequency=motion_info["parameters"]["sampling_frequency"],
            segment_index=segment_index,
            depth_lim=depth_lim,
            motion_lim=motion_lim,
            color_amplitude=color_amplitude,
            scatter_decimate=scatter_decimate,
            amplitude_cmap=amplitude_cmap,
            amplitude_color=amplitude_color,
            amplitude_clim=amplitude_clim,
            amplitude_alpha=amplitude_alpha,
            recording=recording,
            **motion_info,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        from .utils_matplotlib import make_mpl_figure

        from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

        dp = to_attr(data_plot)

        assert backend_kwargs["axes"] is None, "axes argument is not allowed in MotionWidget"
        assert backend_kwargs["ax"] is None, "ax argument is not allowed in MotionWidget"

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        is_rigid = dp.motion.spatial_bins_um.shape[0] == 1

        motion = dp.motion

        displacement = motion.displacement[dp.segment_index]
        temporal_bins_s = motion.temporal_bins_s[dp.segment_index]
        spatial_bins_um = motion.spatial_bins_um

        if dp.motion_lim is None:
            motion_lim = np.max(np.abs(displacement)) * 1.05
        else:
            motion_lim = dp.motion_lim

        is_rigid = displacement.shape[1] == 1

        gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.5)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        if not is_rigid:
            ax3 = fig.add_subplot(gs[1, 1])
        ax1.sharex(ax0)
        ax1.sharey(ax0)

        corrected_location = correct_motion_on_peaks(
            dp.peaks,
            dp.peak_locations,
            dp.motion,
            dp.recording,
        )

        commpon_drift_map_kwargs = dict(
            direction=dp.motion.direction,
            recording=dp.recording,
            segment_index=dp.segment_index,
            depth_lim=dp.depth_lim,
            scatter_decimate=dp.scatter_decimate,
            color_amplitude=dp.color_amplitude,
            color=dp.amplitude_color,
            cmap=dp.amplitude_cmap,
            clim=dp.amplitude_clim,
            alpha=dp.amplitude_alpha,
            backend="matplotlib",
        )

        # with immediate_plot=True the widgets are plotted immediately
        _ = DriftRasterMapWidget(
            dp.peaks,
            dp.peak_locations,
            ax=ax0,
            immediate_plot=True,
            **commpon_drift_map_kwargs,
        )

        _ = DriftRasterMapWidget(
            dp.peaks,
            corrected_location,
            ax=ax1,
            immediate_plot=True,
            **commpon_drift_map_kwargs,
        )

        ax2.plot(temporal_bins_s, displacement, alpha=0.2, color="black")
        ax2.plot(temporal_bins_s, np.mean(displacement, axis=1), color="C0")
        ax2.set_ylim(-motion_lim, motion_lim)
        ax2.set_ylabel("Motion [$\\mu$m]")
        ax2.set_xlabel("Times [s]")
        ax2.set_title("Motion vectors")
        axes = [ax0, ax1, ax2]

        if not is_rigid:
            im = ax3.imshow(
                displacement.T,
                aspect="auto",
                origin="lower",
                extent=(
                    temporal_bins_s[0],
                    temporal_bins_s[-1],
                    spatial_bins_um[0],
                    spatial_bins_um[-1],
                ),
            )
            im.set_clim(-motion_lim, motion_lim)
            cbar = fig.colorbar(im)
            cbar.ax.set_ylabel("Motion [$\\mu$m]")
            ax3.set_xlabel("Times [s]")
            ax3.set_ylabel("Depth [$\\mu$m]")
            ax3.set_title("Motion vectors")
            axes.append(ax3)
        self.axes = np.array(axes)
