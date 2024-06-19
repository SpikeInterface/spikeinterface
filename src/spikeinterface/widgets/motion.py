from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


class MotionWidget(BaseWidget):
    """
    Plot the Motion object

    Parameters
    ----------
    motion : Motion
        The motion object
    segment_index : None | int
        If Motion is multi segment, the must be not None
    mode : "auto" | "line" | "map"
        How to plot map or lines.
        "auto" make it automatic if the number of depth is too high.
    """

    def __init__(
        self,
        motion,
        segment_index=None,
        mode="line",
        motion_lim=None,
        backend=None,
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
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from matplotlib.colors import Normalize

        dp = to_attr(data_plot)

        motion = data_plot["motion"]
        segment_index = data_plot["segment_index"]

        assert backend_kwargs["axes"] is None

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

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


class MotionInfoWidget(BaseWidget):
    """
        Plot motion information from the motion_info dict returned by correct_motion().
        This plot:
          * the motion iself
          * the peak depth vs time before correction
          * the peak depth vs time after correction

    Parameters
    ----------
    motion_info : dict
        The motion info return by correct_motion() or load back with load_motion_info()
    segment_index : int, default: None
        The segment index to display.
    recording : RecordingExtractor, default: None
        The recording extractor object (only used to get "real" times)
    segment_index : int, default: 0
        The segment index to display.
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
    """

    def __init__(
        self,
        motion_info,
        segment_index=None,
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

        motion = motion_info["motion"]
        if segment_index is None:
            if len(motion.displacement) == 1:
                segment_index = 0
            else:
                raise ValueError("plot motion : teh Motion object is multi segment you must provide segmentindex=XX")

        times = recording.get_times() if recording is not None else None

        plot_data = dict(
            sampling_frequency=motion_info["parameters"]["sampling_frequency"],
            times=times,
            segment_index=segment_index,
            depth_lim=depth_lim,
            motion_lim=motion_lim,
            color_amplitude=color_amplitude,
            scatter_decimate=scatter_decimate,
            amplitude_cmap=amplitude_cmap,
            amplitude_clim=amplitude_clim,
            amplitude_alpha=amplitude_alpha,
            recording=recording,
            **motion_info,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from matplotlib.colors import Normalize

        from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

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

        if dp.times is None:
            # temporal_bins_plot = dp.temporal_bins
            x = dp.peaks["sample_index"] / dp.sampling_frequency
        else:
            # use real times and adjust temporal bins with t_start
            # temporal_bins_plot = dp.temporal_bins + dp.times[0]
            x = dp.times[dp.peaks["sample_index"]]

        corrected_location = correct_motion_on_peaks(
            dp.peaks,
            dp.peak_locations,
            dp.motion,
            dp.recording,
        )

        y = dp.peak_locations[motion.direction]
        y2 = corrected_location[motion.direction]
        if dp.scatter_decimate is not None:
            x = x[:: dp.scatter_decimate]
            y = y[:: dp.scatter_decimate]
            y2 = y2[:: dp.scatter_decimate]

        if dp.color_amplitude:
            amps = dp.peaks["amplitude"]
            amps_abs = np.abs(amps)
            q_95 = np.quantile(amps_abs, 0.95)
            if dp.scatter_decimate is not None:
                amps = amps[:: dp.scatter_decimate]
                amps_abs = amps_abs[:: dp.scatter_decimate]
            cmap = plt.colormaps[dp.amplitude_cmap]
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

        ax0.scatter(x, y, s=1, **color_kwargs)
        if dp.depth_lim is not None:
            ax0.set_ylim(*dp.depth_lim)
        ax0.set_title("Peak depth")
        ax0.set_xlabel("Times [s]")
        ax0.set_ylabel("Depth [$\\mu$m]")

        ax1.scatter(x, y2, s=1, **color_kwargs)
        ax1.set_xlabel("Times [s]")
        ax1.set_ylabel("Depth [$\\mu$m]")
        ax1.set_title("Corrected peak depth")

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
