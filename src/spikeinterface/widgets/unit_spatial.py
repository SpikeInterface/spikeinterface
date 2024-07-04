from __future__ import annotations

import numpy as np
from probeinterface import Probe
from probeinterface.plotting import get_auto_lims
from seaborn import color_palette
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
            sorting_analyzer, probe=None,
            depth_axis=1, bins=None,
            cmap="viridis", kde=False,
            depth_hist=True, groups=None,
            backend=None, **backend_kwargs
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        self.check_extensions(sorting_analyzer, "unit_locations")
        ulc = sorting_analyzer.get_extension("unit_locations")
        unit_locations = ulc.get_data(outputs="numpy")
        x, y = unit_locations[:, 0], unit_locations[:, 1]

        if type(probe) is Probe:
            if sorting_analyzer.recording.has_probe():
                # TODO: throw warning saying that sorting_analyzer has a probe and it will be overwritten
                pass
        elif sorting_analyzer.recording.has_probe():
            probe = sorting_analyzer.get_probe()
        else:
            # TODO: throw error or warning, no probe available
            pass
        
        xrange, yrange, _ = get_auto_lims(probe, margin=0)
        if bins is None:
            bins = (
                np.round(np.diff(xrange).squeeze() / 75).astype(int),
                np.round(np.diff(yrange).squeeze() / 75).astype(int)
            )

        if type(cmap) is str:
            cmap = color_palette(cmap, as_cmap=True)

        plot_data = dict(
            probe=probe,
            x=x,
            y=y,
            depth_axis=depth_axis,
            xrange=xrange,
            yrange=yrange,
            bins=bins,
            kde=kde,
            cmap=cmap,
            depth_hist=depth_hist,
            groups=groups
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.patches as patches
        import matplotlib.path as path
        from seaborn import kdeplot, histplot
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        ax = self.ax

        custom_shape = path.Path(dp.probe.probe_planar_contour)
        patch = patches.PathPatch(custom_shape, facecolor="none", edgecolor="none")
        ax.add_patch(patch)

        if dp.kde is not True:
            hist, xedges, yedges = np.histogram2d(dp.x, dp.y, bins=dp.bins, range=[dp.xrange, dp.yrange])
            pcm = ax.pcolormesh(xedges, yedges, hist.T, cmap=dp.cmap)
        else:
            data = dict(x=dp.x, y=dp.y)
            bg = ax.add_patch(
                patches.Rectangle(
                    [dp.xrange[0], dp.yrange[0]],
                    np.diff(dp.xrange).squeeze(),
                    np.diff(dp.yrange).squeeze(),
                    facecolor=dp.cmap.colors[0],
                    fill=True
                )
            )
            bg.set_clip_path(patch)
            kdeplot(
                data, x='x', y='y',
                cmap=dp.cmap, levels=100, thresh=0, fill=True,
                ax=ax, bw_adjust=0.1, clip=[dp.xrange, dp.yrange]
            )
            pcm = ax.collections[0]
            ax.set_xlabel(None)
            ax.set_ylabel(None)

        pcm.set_clip_path(patch)

        xlim, ylim, _ = get_auto_lims(dp.probe, margin=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('Depth (um)')

        if dp.depth_hist is True:
            bbox = ax.get_window_extent()
            hist_height = 1.5 * bbox.width

            ax_hist = ax.inset_axes([1, 0, hist_height / bbox.width, 1])
            data = dict(y=dp.y)
            data['group'] = np.ones(dp.y.size) if dp.groups is None else dp.groups
            palette = color_palette('bright', n_colors=1 if dp.groups is None else np.unique(dp.groups).size)
            histplot(data=data, y='y', hue='group', bins=dp.bins[1], binrange=dp.yrange, palette=palette, ax=ax_hist, legend=False)
            ax_hist.axis('off')
            ax_hist.set_ylim(*ylim)