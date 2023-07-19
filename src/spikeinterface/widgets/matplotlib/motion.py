from ..base import to_attr
from ..motion import MotionWidget
from .base_mpl import MplPlotter

import numpy as np
from matplotlib.colors import Normalize


class MotionPlotter(MplPlotter):
    def do_plot(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        assert backend_kwargs["axes"] is None
        assert backend_kwargs["ax"] is None

        self.make_mpl_figure(**backend_kwargs)
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

        if dp.motion_lim is None:
            motion_lim = np.max(np.abs(dp.motion)) * 1.05
        else:
            motion_lim = dp.motion_lim

        if dp.times is None:
            temporal_bins_plot = dp.temporal_bins
            x = dp.peaks["sample_index"] / dp.sampling_frequency
        else:
            # use real times and adjust temporal bins with t_start
            temporal_bins_plot = dp.temporal_bins + dp.times[0]
            x = dp.times[dp.peaks["sample_index"]]

        corrected_location = correct_motion_on_peaks(
            dp.peaks,
            dp.peak_locations,
            dp.sampling_frequency,
            dp.motion,
            dp.temporal_bins,
            dp.spatial_bins,
            direction="y",
        )

        y = dp.peak_locations["y"]
        y2 = corrected_location["y"]
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

        ax0.scatter(x, y, s=1, **color_kwargs)
        if dp.depth_lim is not None:
            ax0.set_ylim(*dp.depth_lim)
        ax0.set_title("Peak depth")
        ax0.set_xlabel("Times [s]")
        ax0.set_ylabel("Depth [um]")

        ax1.scatter(x, y2, s=1, **color_kwargs)
        ax1.set_xlabel("Times [s]")
        ax1.set_ylabel("Depth [um]")
        ax1.set_title("Corrected peak depth")

        ax2.plot(temporal_bins_plot, dp.motion, alpha=0.2, color="black")
        ax2.plot(temporal_bins_plot, np.mean(dp.motion, axis=1), color="C0")
        ax2.set_ylim(-motion_lim, motion_lim)
        ax2.set_ylabel("Motion [um]")
        ax2.set_title("Motion vectors")
        axes = [ax0, ax1, ax2]

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
            im.set_clim(-motion_lim, motion_lim)
            cbar = fig.colorbar(im)
            cbar.ax.set_xlabel("motion [um]")
            ax3.set_xlabel("Times [s]")
            ax3.set_ylabel("Depth [um]")
            ax3.set_title("Motion vectors")
            axes.append(ax3)
        self.axes = np.array(axes)


MotionPlotter.register(MotionWidget)
