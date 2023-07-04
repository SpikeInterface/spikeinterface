from ..base import to_attr
from ..motion import MotionWidget
from .base_mpl import MplPlotter

import numpy as np


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

        corrected_location = correct_motion_on_peaks(
            dp.peaks, dp.peak_locations, dp.rec.get_times(), dp.motion, dp.temporal_bins, dp.spatial_bins, direction="y"
        )

        x = dp.peaks["sample_index"] / dp.rec.get_sampling_frequency()
        y = dp.peak_locations["y"]
        y2 = corrected_location["y"]
        if dp.scatter_decimate is not None:
            x = x[:: dp.scatter_decimate]
            y = y[:: dp.scatter_decimate]
            y2 = y2[:: dp.scatter_decimate]

        if dp.color_amplitude:
            amps = np.abs(dp.peaks["amplitude"])
            amps /= np.quantile(amps, 0.95)
            if dp.scatter_decimate is not None:
                amps = amps[:: dp.scatter_decimate]
            c = plt.get_cmap(dp.amplitude_cmap)(amps)
            color_kwargs = dict(
                color=None,
                c=c,
            )  # alpha=0.02
        else:
            color_kwargs = dict(color="k", c=None)  # alpha=0.02

        ax0.scatter(x, y, s=1, **color_kwargs)
        # for i in range(dp.motion.shape[1]):
        #     ax0.plot(dp.temporal_bins, dp.motion[:, i] + dp.spatial_bins[i], color="C8", alpha=1.0)
        if dp.depth_lim is not None:
            ax0.set_ylim(*dp.depth_lim)
        ax0.set_title("Peak depth")
        ax0.set_xlabel("Times [s]")
        ax0.set_ylabel("Depth [um]")

        ax1.scatter(x, y2, s=1, **color_kwargs)
        ax1.set_xlabel("Times [s]")
        ax1.set_ylabel("Depth [um]")
        ax1.set_title("Corrected peak depth")

        ax2.plot(dp.temporal_bins, dp.motion, alpha=0.2, color="black")
        ax2.plot(dp.temporal_bins, np.mean(dp.motion, axis=1), color="C0")
        ax2.set_ylim(-motion_lim, motion_lim)
        ax2.set_ylabel("motion [um]")
        ax2.set_title("Motion vectors")

        if not is_rigid:
            im = ax3.imshow(
                dp.motion.T,
                aspect="auto",
                origin="lower",
                extent=(
                    dp.temporal_bins[0],
                    dp.temporal_bins[-1],
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


MotionPlotter.register(MotionWidget)
