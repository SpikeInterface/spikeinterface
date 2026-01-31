from __future__ import annotations

import numpy as np
from probeinterface import Probe
from warnings import warn
from .base import BaseWidget, to_attr


class UnitSpatialDistributionsWidget(BaseWidget):
    """
    Placeholder documentation to be changed.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    depth_axis : int, default: 1
        The dimension of unit_locations that is depth
    """

    def __init__(
        self,
        sorting_analyzer,
        probe=None,
        depth_axis=1,
        bins=None,
        cmap="viridis",
        kde=False,
        depth_hist=True,
        groups=None,
        kde_kws=None,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        self.check_extensions(sorting_analyzer, "unit_locations")
        ulc = sorting_analyzer.get_extension("unit_locations")
        unit_locations = ulc.get_data(outputs="numpy")
        x, y = unit_locations[:, 0], unit_locations[:, 1]

        if type(probe) is Probe:
            if sorting_analyzer.recording.has_probe():
                warn(
                    "There is a Probe attached to this recording, but the probe argument is not None: the attached Probe will be ignored."
                )
        elif sorting_analyzer.recording.has_probe():
            probe = sorting_analyzer.get_probe()
        else:
            raise ValueError(
                "There is no Probe attached to this recording. Use set_probe(...) to attach one or pass it to the function via the probe argument."
            )

        # xrange, yrange, _ = get_auto_lims(probe, margin=0)
        # if bins is None:
        #     bins = (
        #         np.round(np.diff(xrange).squeeze() / 75).astype(int),
        #         np.round(np.diff(yrange).squeeze() / 75).astype(int),
        #     )
        #     # TODO: change behaviour, if bins is not defined, bin only along the depth axis

        plot_data = dict(
            probe=probe,
            x=x,
            y=y,
            depth_axis=depth_axis,
            bins=bins,
            kde=kde,
            cmap=cmap,
            depth_hist=depth_hist,
            groups=groups,
            kde_kws=kde_kws,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.patches as patches
        import matplotlib.path as path
        from probeinterface.plotting import get_auto_lims
        from seaborn import color_palette, kdeplot, histplot
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        xrange, yrange, _ = get_auto_lims(dp.probe, margin=0)
        cmap = color_palette(dp.cmap, as_cmap=True) if type(dp.cmap) is str else dp.cmap

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        ax = self.ax

        custom_shape = path.Path(dp.probe.probe_planar_contour)
        patch = patches.PathPatch(custom_shape, facecolor="none", edgecolor="none")
        ax.add_patch(patch)

        if dp.kde is not True:
            hist, xedges, yedges = np.histogram2d(dp.x, dp.y, bins=dp.bins, range=[xrange, yrange])
            pcm = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap)
        else:
            kde_kws = dict(levels=100, thresh=0, fill=True, bw_adjust=0.1)
            if dp.kde_kws is not None:
                kde_kws.update(dp.kde_kws)
            data = dict(x=dp.x, y=dp.y)
            bg = ax.add_patch(
                patches.Rectangle(
                    [xrange[0], yrange[0]],
                    np.diff(xrange).squeeze(),
                    np.diff(yrange).squeeze(),
                    facecolor=cmap.colors[0],
                    fill=True,
                )
            )
            bg.set_clip_path(patch)
            kdeplot(data, x="x", y="y", clip=[xrange, yrange], cmap=cmap, ax=ax, **kde_kws)
            pcm = ax.collections[0]
            ax.set_xlabel(None)
            ax.set_ylabel(None)

        pcm.set_clip_path(patch)

        xlim, ylim, _ = get_auto_lims(dp.probe, margin=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel("Depth (um)")

        if dp.depth_hist is True:
            bbox = ax.get_window_extent()
            hist_height = 1.5 * bbox.width

            ax_hist = ax.inset_axes([1, 0, hist_height / bbox.width, 1])
            data = dict(y=dp.y)
            data["group"] = np.ones(dp.y.size) if dp.groups is None else dp.groups
            palette = color_palette("bright", n_colors=1 if dp.groups is None else np.unique(dp.groups).size)
            histplot(
                data=data,
                y="y",
                hue="group",
                bins=dp.bins[1],
                binrange=yrange,
                palette=palette,
                ax=ax_hist,
                legend=False,
            )
            ax_hist.axis("off")
            ax_hist.set_ylim(*ylim)
