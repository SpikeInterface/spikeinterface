from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from probeinterface import Probe
from .base import BaseWidget, to_attr

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from numpy.typing import ArrayLike
    from spikeinterface.core import SortingAnalyzer
    from spikeinterface.core.waveforms_extractor_backwards_compatibility import MockWaveformExtractor


class UnitSpatialDistributionsWidget(BaseWidget):
    """
    Plot unit spatial distributions as a histogram or kernel density estimate.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        Analyzer with the unit_locations extension already computed.
    probe : Probe or None, default: None
        Probe geometry to display. If None, use the analyzer's stored probe.
    depth_axis : int, default: 1
        Coordinate along the probe representing depth: 0 for x or 1 for y.
    bins : int, tuple or None, default: None
        Bin count for both axes, or a pair of counts or bin-edge arrays for x
        and y. If None, use 75 um per depth bin.
    cmap : str or Colormap, default: "viridis"
        Colormap name or object for the spatial histogram or KDE.
    kde : bool, default: False
        If True, show a two-dimensional kernel density estimate instead of
        the spatial histogram.
    depth_hist : bool, default: True
        Show a depth histogram to the right for y depth or below for x depth.
    groups : array-like or None, default: None
        Group label for each unit, in analyzer unit order. Colors the depth
        histogram by group, e.g. ["good", "good", "bad", "bad"] for four units.
        If None, use one group for all units.
        histogram by group. If None, use one group for all units.
    kde_kwargs : dict or None, default: None
        Keyword arguments passed to seaborn.kdeplot, overriding the defaults.
        Only used for KDE.
    show_depth_hist_axis : bool, default: True
        Show the count-axis on the depth histogram.
        Only used when depth_hist is True.
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        probe: Probe | None = None,
        depth_axis: Literal[0, 1] = 1,
        bins: int | tuple | None = None,
        cmap: str | Colormap = "viridis",
        kde: bool = False,
        depth_hist: bool = True,
        groups: ArrayLike | None = None,
        kde_kwargs: dict[str, Any] | None = None,
        backend: str | None = None,
        show_depth_hist_axis: bool = True,
        **backend_kwargs: Any,
    ) -> None:
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        self.check_extensions(sorting_analyzer, "unit_locations")
        ulc = sorting_analyzer.get_extension("unit_locations")
        unit_locations = ulc.get_data(outputs="numpy")

        if depth_axis not in (0, 1):
            raise ValueError("depth_axis must be 0 (x) or 1 (y)")

        x, y = unit_locations[:, 0], unit_locations[:, 1]

        if probe is None:
            probe = sorting_analyzer.get_probe()
        elif not isinstance(probe, Probe):
            raise TypeError("probe must be a Probe or None")

        plot_data = dict(
            probe=probe,
            x=x,
            y=y,
            depth_axis=depth_axis,
            bins=bins,
            kde=kde,
            cmap=cmap,
            depth_hist=depth_hist,
            show_depth_hist_axis=show_depth_hist_axis,
            groups=groups,
            kde_kwargs=kde_kwargs,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot: dict[str, Any], **backend_kwargs: Any) -> None:
        import matplotlib.patches as patches
        import matplotlib.path as mpl_paths
        from probeinterface.plotting import get_auto_lims
        from seaborn import color_palette, kdeplot, histplot
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        xrange, yrange, _ = get_auto_lims(dp.probe, margin=0)
        bins = dp.bins

        if bins is None:
            # if bins not supplied, divive the probe into 75 um bins
            depth_range = (xrange, yrange)[dp.depth_axis]
            bins = [1, 1]
            bins[dp.depth_axis] = max(1, int(np.ceil(np.ptp(depth_range) / 75.0)))

        # Explicitly create the depth bin edges so they can be
        # shared between the spatial and marginal histograms.
        bin_specs = (bins, bins) if np.isscalar(bins) else bins
        bins = (
            np.histogram_bin_edges(dp.x, bins=bin_specs[0], range=xrange),
            np.histogram_bin_edges(dp.y, bins=bin_specs[1], range=yrange),
        )
        cmap = color_palette(dp.cmap, as_cmap=True) if type(dp.cmap) is str else dp.cmap

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        ax = self.ax
        if dp.depth_hist and backend_kwargs.get("ax") is None and backend_kwargs.get("axes") is None:
            # Keep the marginal histogram inside the default figure bounds.
            # Fixed layout proportions reserve space for the depth histogram.
            position = ax.get_position()
            if dp.depth_axis == 1:
                ax.set_position([position.x0, position.y0, position.width / 2.5, position.height])
            else:
                height = position.height / 1.6
                ax.set_position([position.x0, position.y0 + 0.6 * height, position.width, height])

        # Set the clipping region for the spatial plot based on the probe dimensions
        probe_outline = None
        if dp.probe.probe_planar_contour is not None:
            probe_outline_path = mpl_paths.Path(dp.probe.probe_planar_contour)
            probe_outline = patches.PathPatch(probe_outline_path, facecolor="none", edgecolor="none")
            ax.add_patch(probe_outline)

        # Create the spatial plot
        if dp.kde is not True:
            hist, xedges, yedges = np.histogram2d(dp.x, dp.y, bins=bins, range=[xrange, yrange])
            spatial_plot = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap)
        else:
            kde_kwargs = dict(levels=100, thresh=0, fill=True, bw_adjust=0.1)
            if dp.kde_kwargs is not None:
                kde_kwargs.update(dp.kde_kwargs)

            data = dict(x=dp.x, y=dp.y)
            kde_background = ax.add_patch(
                patches.Rectangle(
                    [xrange[0], yrange[0]],
                    np.diff(xrange).squeeze(),
                    np.diff(yrange).squeeze(),
                    facecolor=cmap(0.0),
                    fill=True,
                )
            )
            if probe_outline is not None:
                kde_background.set_clip_path(probe_outline)

            kdeplot(data, x="x", y="y", clip=[xrange, yrange], cmap=cmap, ax=ax, **kde_kwargs)
            spatial_plot = ax.collections[0]
            ax.set_xlabel(None)
            ax.set_ylabel(None)

        if probe_outline is not None:
            spatial_plot.set_clip_path(probe_outline)

        # Create the histogram plot
        xlim, ylim, _ = get_auto_lims(dp.probe, margin=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if dp.depth_axis == 1:
            ax.set_xticks([])
            ax.set_xlabel("")
            ax.set_ylabel("Depth (um)")
        else:
            ax.set_yticks([])
            ax.set_xlabel("Depth (um)")
            ax.set_ylabel("")

        if dp.depth_hist is True:
            bbox = ax.get_window_extent()
            hist_height = 1.5 * bbox.width

            if dp.depth_axis == 1:
                ax_hist = ax.inset_axes([1, 0, hist_height / bbox.width, 1])
                depth_key = "y"
                depth_values = dp.y
            else:
                ax_hist = ax.inset_axes([0, -0.6, 1, 0.4])
                depth_key = "x"
                depth_values = dp.x

            data = {depth_key: depth_values}
            data["group"] = np.ones(dp.y.size) if dp.groups is None else dp.groups
            palette = color_palette("bright", n_colors=1 if dp.groups is None else np.unique(dp.groups).size)

            histplot(
                data=data,
                **{depth_key: depth_key},
                hue="group",
                bins=bins[dp.depth_axis],
                binrange=(xrange, yrange)[dp.depth_axis],
                palette=palette,
                ax=ax_hist,
                legend=False,
            )

            if dp.show_depth_hist_axis:
                # Show axis for counts on histogram
                if dp.depth_axis == 1:
                    ax_hist.yaxis.set_visible(False)
                    ax_hist.set_xlabel("Count")
                    visible_spine = "bottom"
                else:
                    ax_hist.xaxis.set_visible(False)
                    ax_hist.set_ylabel("Count")
                    visible_spine = "left"

                for name, spine in ax_hist.spines.items():
                    spine.set_visible(name == visible_spine)
            else:
                ax_hist.axis("off")

            if dp.depth_axis == 1:
                ax_hist.set_ylim(*ylim)
            else:
                ax_hist.set_xlim(*xlim)
