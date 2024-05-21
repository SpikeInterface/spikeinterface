from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


class MotionWidget(BaseWidget):
    """
    Plot a summary of motion correction outputs. The plots shown are:

    1) 'Peak depth' plot displaying the peak location for all detected spikes over time
       before motion correction.
    2) 'Corrected peak depth' displays the peak location for all detected spikes
       over time after motion correction.
    3) 'Motion vectors' plot shows the estimated displacement per spatial
        bin across time. The average displacement is shown in blue.
    4) 'Motion vectors nonrigid' plot shows a heatmap of the estimated
       motion across the depth of the probe and time.

    Parameters
    ----------
    motion_info: dict
        The motion info return by correct_motion() or load back with load_motion_info()
    recording : RecordingExtractor, default: None
        The recording extractor object (only used to get "real" times)
    sampling_frequency : float, default: None
        The sampling frequency (needed if recording is None)
    depth_lim : tuple or None, default: None
        The min and max depth to display, if None (min and max of the recording)
    motion_lim : tuple or None, default: None
        The min and max motion to display, if None (min and max of the motion)
    color_amplitude : bool, default: False
        If True, the color of the scatter points is the amplitude of the peaks
    scatter_decimate : int, default: None
        If > 1, the scatter points are decimated
    amplitude_cmap : str, default: "inferno"
        The colormap to use for the amplitude
    amplitude_clim : tuple or None, default: None
        The min and max amplitude to display, if None (min and max of the amplitudes)
    amplitude_alpha : float, default: 1
        The alpha of the scatter points
    backend : Union[None, "matplotlib"]
        Determines the plotting backend. Only "matplotlib" currently supported.
    """

    def __init__(
        self,
        motion_info,
        recording=None,
        depth_lim=None,
        motion_lim=None,
        color_amplitude=False,
        scatter_decimate=None,
        amplitude_cmap="inferno",
        amplitude_clim=None,
        amplitude_alpha=1,
        backend=None,
        **backend_kwargs,
    ):
        times = recording.get_times() if recording is not None else None

        plot_data = dict(
            sampling_frequency=motion_info["parameters"]["sampling_frequency"],
            times=times,
            depth_lim=depth_lim,
            motion_lim=motion_lim,
            color_amplitude=color_amplitude,
            scatter_decimate=scatter_decimate,
            amplitude_cmap=amplitude_cmap,
            amplitude_clim=amplitude_clim,
            amplitude_alpha=amplitude_alpha,
            **motion_info,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        """
        Plot the motion correction plot using the
        "matplotlib" backend.
        """
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from matplotlib.colors import Normalize
        from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

        dp = to_attr(data_plot)

        if not (backend_kwargs["axes"] is None and backend_kwargs["ax"] is None):
            raise ValueError(
                "`ax` and `axes` arguments are not permitted "
                "for `plot_motion` as multiple axes are required. "
                "Instead supply the `fig` parameter to plot on"
                "a specific object."
            )

        # Setup figures and axes ----------------------------------------------

        self.ax = None
        self.figure, self.axes, _ = make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        is_rigid = dp.motion.shape[1] == 1

        gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.3)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        if not is_rigid:
            ax3 = fig.add_subplot(gs[1, 1])
        ax1.sharex(ax0)
        ax1.sharey(ax0)

        # Plot the peak depth and corrected peak depth ------------------------

        if dp.times is None:
            temporal_bins_plot = dp.temporal_bins
            peak_times = dp.peaks["sample_index"] / dp.sampling_frequency
        else:
            # use real times and adjust temporal bins with t_start
            temporal_bins_plot = dp.temporal_bins + dp.times[0]
            peak_times = dp.times[dp.peaks["sample_index"]]

        corrected_location = correct_motion_on_peaks(
            dp.peaks,
            dp.peak_locations,
            dp.sampling_frequency,
            dp.motion,
            dp.temporal_bins,
            dp.spatial_bins,
            direction="y",
        )

        peak_locs = dp.peak_locations["y"]
        corr_peak_locs = corrected_location["y"]

        if dp.scatter_decimate is not None:
            peak_times = peak_times[:: dp.scatter_decimate]
            peak_locs = peak_locs[:: dp.scatter_decimate]
            corr_peak_locs = corr_peak_locs[:: dp.scatter_decimate]

        if dp.color_amplitude:
            amps = dp.peaks["amplitude"]
            amps_abs = np.abs(amps)
            q_95 = np.quantile(amps_abs, 0.95)
            if dp.scatter_decimate is not None:
                amps = amps[:: dp.scatter_decimate]
                amps_abs = amps_abs[:: dp.scatter_decimate]
            cmap = plt.get_cmap(dp.amplitude_cmap)
            if dp.amplitude_clim is None:
                amps = amps_abs
                amps /= q_95
                c = cmap(amps)
            else:
                norm_function = Normalize(vmin=dp.amplitude_clim[0], vmax=dp.amplitude_clim[1], clip=True)
                c = cmap(norm_function(amps))
            color_kwargs = dict(
                color=None,
                c=c,
                alpha=dp.amplitude_alpha,
            )
        else:
            color_kwargs = dict(color="k", c=None, alpha=dp.amplitude_alpha)

        ax0.scatter(peak_times, peak_locs, s=1, **color_kwargs)
        if dp.depth_lim is not None:
            ax0.set_ylim(*dp.depth_lim)
        ax0.set_title("Peak depth")
        ax0.set_xlabel("Time [s]")
        ax0.set_ylabel("Depth [μm]")

        ax1.scatter(peak_times, corr_peak_locs, s=1, **color_kwargs)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Depth [μm]")
        ax1.set_title("Corrected peak depth")

        # Plot the Motion Vectors ---------------------------------------------

        if dp.motion_lim is None:
            max_motion_lim = np.max(np.abs(dp.motion)) * 1.05
            motion_lim = (-max_motion_lim, max_motion_lim)
        else:
            motion_lim = dp.motion_lim

        ax2.plot(temporal_bins_plot, dp.motion, alpha=0.2, color="black")
        ax2.plot(temporal_bins_plot, np.mean(dp.motion, axis=1), color="C0")
        ax2.set_ylim(motion_lim)
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Motion [μm]")
        ax2.set_title("Motion vectors")
        axes = [ax0, ax1, ax2]

        # Plot the motion-by-depth colormap (if not rigid) --------------------

        if not is_rigid:
            im = ax3.imshow(
                dp.motion.T,
                aspect="auto",
                origin="lower",
                extent=(
                    temporal_bins_plot[0],
                    temporal_bins_plot[-1],
                    dp.spatial_bins[0],
                    dp.spatial_bins[-1],
                ),
            )
            im.set_clim(motion_lim)
            cbar = fig.colorbar(im)
            cbar.ax.set_xlabel("motion [μm]")
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("Depth [μm]")
            ax3.set_title("Motion vectors")
            axes.append(ax3)

        self.axes = np.array(axes)
