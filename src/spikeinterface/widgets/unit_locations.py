from __future__ import annotations

from probeinterface import ProbeGroup

from .base import BaseWidget, to_attr
from .utils import get_unit_colors
from spikeinterface.core.sortinganalyzer import SortingAnalyzer
import numpy as np


class UnitLocationsWidget(BaseWidget):
    """
    Plots each unit's location.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer that must contains  "unit_locations" extension
    unit_ids : list or None, default: None
        List of unit ids
    with_channel_ids : bool, default: False
        Add channel ids text on the probe
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
    hide_unit_selector : bool, default: False
        If True, the unit selector is not displayed (sortingview backend)
    plot_all_units : bool, default: True
        If True, all units are plotted. The unselected ones (not in unit_ids),
        are plotted in grey (matplotlib backend)
    plot_legend : bool, default: False
        If True, the legend is plotted (matplotlib backend)
    hide_axis : bool, default: False
        If True, the axis is set to off (matplotlib backend)
    margin : float, default: 50
        Amount of margin to add to plot, beyond the extremum unit locations.
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids: list | None = None,
        with_channel_ids: bool = False,
        unit_colors: dict | None = None,
        hide_unit_selector: bool = False,
        plot_all_units: bool = True,
        plot_legend: bool = False,
        hide_axis: bool = False,
        backend: str | None = None,
        margin: float = 50,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        self.check_extensions(sorting_analyzer, "unit_locations")
        ulc = sorting_analyzer.get_extension("unit_locations")
        unit_locations = ulc.get_data(outputs="by_unit")

        # set axis limits based on extremum unit locations
        all_unit_locations = ulc.get_data()

        x_locations = all_unit_locations[:, 0]
        x_min = np.min(x_locations)
        x_max = np.max(x_locations)
        x_lim = (x_min - margin, x_max + margin)

        y_locations = all_unit_locations[:, 1]
        y_min = np.min(y_locations)
        y_max = np.max(y_locations)
        y_lim = (y_min - margin, y_max + margin)

        sorting = sorting_analyzer.sorting

        channel_ids = sorting_analyzer.channel_ids
        channel_locations = sorting_analyzer.get_channel_locations()
        probegroup = sorting_analyzer.get_probegroup()

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        data_plot = dict(
            all_unit_ids=sorting.unit_ids,
            unit_locations=unit_locations,
            sorting=sorting,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            probegroup_dict=probegroup.to_dict(),
            with_channel_ids=with_channel_ids,
            hide_unit_selector=hide_unit_selector,
            plot_all_units=plot_all_units,
            plot_legend=plot_legend,
            hide_axis=hide_axis,
            x_lim=x_lim,
            y_lim=y_lim,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from probeinterface.plotting import plot_probe

        from matplotlib.patches import Ellipse
        from matplotlib.lines import Line2D

        dp = to_attr(data_plot)
        # backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        # self.make_mpl_figure(**backend_kwargs)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        unit_locations = dp.unit_locations

        probegroup = ProbeGroup.from_dict(dp.probegroup_dict)
        probe_shape_kwargs = dict(facecolor="w", edgecolor="k", lw=0.5, alpha=1.0)
        contacts_kargs = dict(alpha=1.0, edgecolor="k", lw=0.5)

        for probe in probegroup.probes:
            text_on_contact = None
            if dp.with_channel_ids:
                text_on_contact = dp.channel_ids

            poly_contact, poly_contour = plot_probe(
                probe,
                ax=self.ax,
                contacts_colors="w",
                contacts_kargs=contacts_kargs,
                probe_shape_kwargs=probe_shape_kwargs,
                text_on_contact=text_on_contact,
            )
            poly_contact.set_zorder(2)
            if poly_contour is not None:
                poly_contour.set_zorder(1)

        self.ax.set_title("")
        self.ax.set_xlim(*dp.x_lim)
        self.ax.set_ylim(*dp.y_lim)

        width = height = 10
        ellipse_kwargs = dict(width=width, height=height, lw=2)

        if dp.plot_all_units:
            unit_colors = {}
            unit_ids = dp.all_unit_ids
            for unit_id in dp.all_unit_ids:
                if unit_id not in dp.unit_ids:
                    unit_colors[unit_id] = "gray"
                else:
                    unit_colors[unit_id] = dp.unit_colors[unit_id]
        else:
            unit_ids = dp.unit_ids
            unit_colors = dp.unit_colors
        labels = dp.unit_ids

        patches = [
            Ellipse(
                (unit_locations[unit_id]),
                color=unit_colors[unit_id],
                zorder=5 if unit_id in dp.unit_ids else 3,
                alpha=0.9 if unit_id in dp.unit_ids else 0.5,
                **ellipse_kwargs,
            )
            for unit_ind, unit_id in enumerate(unit_ids)
        ]
        for p in patches:
            self.ax.add_patch(p)
        handles = [
            Line2D([0], [0], ls="", marker="o", markersize=5, markeredgewidth=2, color=unit_colors[unit])
            for unit in dp.unit_ids
        ]

        if dp.plot_legend:
            if hasattr(self, "legend") and self.legend is not None:
                # if self.legend is not None:
                self.legend.remove()
            self.legend = self.figure.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True, shadow=True
            )

        if dp.hide_axis:
            self.ax.axis("off")

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()

        cm = 1 / 2.54

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = widgets.Output()
            with output:
                fig, self.ax = plt.subplots(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        self.unit_selector = UnitSelector(data_plot["unit_ids"])
        self.unit_selector.value = list(data_plot["unit_ids"])[:1]

        self.widget = widgets.AppLayout(
            center=fig.canvas,
            left_sidebar=self.unit_selector,
            pane_widths=ratios + [0],
        )

        # a first update
        self._update_ipywidget()

        self.unit_selector.observe(self._update_ipywidget, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _update_ipywidget(self, change=None):
        self.ax.clear()

        # matplotlib next_data_plot dict update at each call
        data_plot = self.next_data_plot
        data_plot["unit_ids"] = self.unit_selector.value
        data_plot["plot_all_units"] = True
        # TODO later add an option checkbox for legend
        data_plot["plot_legend"] = True
        data_plot["hide_axis"] = True

        backend_kwargs = dict(ax=self.ax)

        self.plot_matplotlib(data_plot, **backend_kwargs)
        fig = self.ax.get_figure()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import generate_unit_table_view, make_serializable, handle_display_and_url

        dp = to_attr(data_plot)

        # ensure serializable for sortingview
        unit_ids, channel_ids = make_serializable(dp.unit_ids, dp.channel_ids)

        locations = {str(ch): dp.channel_locations[i_ch].astype("float32") for i_ch, ch in enumerate(channel_ids)}

        unit_items = []
        for unit_id in unit_ids:
            unit_items.append(
                vv.UnitLocationsItem(
                    unit_id=unit_id, x=float(dp.unit_locations[unit_id][0]), y=float(dp.unit_locations[unit_id][1])
                )
            )

        v_unit_locations = vv.UnitLocations(units=unit_items, channel_locations=locations, disable_auto_rotate=True)

        if not dp.hide_unit_selector:
            v_units_table = generate_unit_table_view(dp.sorting)

            self.view = vv.Box(
                direction="horizontal",
                items=[vv.LayoutItem(v_units_table, max_size=150), vv.LayoutItem(v_unit_locations)],
            )
        else:
            self.view = v_unit_locations

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)
