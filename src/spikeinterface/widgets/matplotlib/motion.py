from ..base import to_attr
from ..motion import MotionWidget
from .base_mpl import MplPlotter

import numpy as np


class MotionPlotter(MplPlotter):
    def do_plot(self, data_plot, **backend_kwargs):
        from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks

        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        assert backend_kwargs["axes"] is None
        assert backend_kwargs["ax"] is None

        self.make_mpl_figure(**backend_kwargs)
        fig = self.figure
        fig.clear()

        is_rigid = dp.motion.shape[1] == 1

        gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.1)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        if not is_rigid:
            ax3 = fig.add_subplot(gs[1, 1])
        ax0.sharex(ax1)

        # run_times = motion_info['run_times']
        # peaks = motion_info['peaks']
        # peak_locations = motion_info['peak_locations']
        # temporal_bins = motion_info['temporal_bins']
        # spatial_bins = motion_info['spatial_bins']
        # temporal_bins = motion_info['temporal_bins']
        # motion = motion_info['motion']

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

        ax0.scatter(x, y, s=1, color="k", alpha=0.02)
        for i in range(dp.motion.shape[1]):
            ax0.plot(dp.temporal_bins, dp.motion[:, i] + dp.spatial_bins[i], color="C3", alpha=0.4)

        if dp.depth_lim is not None:
            ax0.set_ylim(*dp.depth_lim)

        ax1.scatter(x, y2, s=1, color="k", alpha=0.02)

        ax2.plot(dp.motion, alpha=0.2, color="black")
        ax2.plot(np.mean(dp.motion, axis=1), color="C0")
        ax2.set_ylim(-motion_lim, motion_lim)
        ax2.set_ylabel("motion [um]")

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
            ax3.set_xlabel("times [s]")
            ax3.set_ylabel("depth [um]")


MotionPlotter.register(MotionWidget)
