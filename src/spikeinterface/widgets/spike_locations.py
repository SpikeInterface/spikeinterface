from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr
from .utils import get_unit_colors
from spikeinterface.core.sortinganalyzer import SortingAnalyzer


class SpikeLocationsWidget(BaseWidget):
    """
    Plots spike locations.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The object to get spike locations from
    unit_ids : list or None, default: None
        List of unit ids
    segment_index : int or None, default: None
        The segment index (or None if mono-segment)
    max_spikes_per_unit : int or None, default: 500
        Number of max spikes per unit to display. Use None for all spikes.
    with_channel_ids : bool, default: False
        Add channel ids text on the probe
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
    hide_unit_selector : bool, default: False
        For sortingview backend, if True the unit selector is not displayed
    plot_all_units : bool, default: True
        If True, all units are plotted. The unselected ones (not in unit_ids),
        are plotted in grey (matplotlib backend)
    plot_legend : bool, default: False
        If True, the legend is plotted (matplotlib backend)
    hide_axis : bool, default: False
        If True, the axis is set to off (matplotlib backend)
    """

    # possible_backends = {}

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids=None,
        segment_index=None,
        max_spikes_per_unit=500,
        with_channel_ids=False,
        unit_colors=None,
        hide_unit_selector=False,
        plot_all_units=True,
        plot_legend=False,
        hide_axis=False,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(sorting_analyzer, "spike_locations")

        spike_locations_by_units = sorting_analyzer.get_extension("spike_locations").get_data(outputs="by_unit")

        sorting = sorting_analyzer.sorting

        channel_ids = sorting_analyzer.channel_ids
        channel_locations = sorting_analyzer.get_channel_locations()
        probegroup = sorting_analyzer.get_probegroup()

        if sorting.get_num_segments() > 1:
            assert segment_index is not None, "Specify segment index for multi-segment object"
        else:
            segment_index = 0

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        all_spike_locs = spike_locations_by_units[segment_index]
        if max_spikes_per_unit is None:
            spike_locs = all_spike_locs
        else:
            spike_locs = dict()
            for unit, locs_unit in all_spike_locs.items():
                if len(locs_unit) > max_spikes_per_unit:
                    random_idxs = np.random.choice(len(locs_unit), size=max_spikes_per_unit, replace=False)
                    spike_locs[unit] = locs_unit[random_idxs]
                else:
                    spike_locs[unit] = locs_unit

        plot_data = dict(
            sorting=sorting,
            all_unit_ids=sorting.unit_ids,
            spike_locations=spike_locs,
            segment_index=segment_index,
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
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from matplotlib.lines import Line2D

        from probeinterface import ProbeGroup
        from probeinterface.plotting import plot_probe

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        spike_locations = dp.spike_locations

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

        if dp.plot_all_units:
            unit_colors = {}
            unit_ids = dp.all_unit_ids
            for unit in dp.all_unit_ids:
                if unit not in dp.unit_ids:
                    unit_colors[unit] = "gray"
                else:
                    unit_colors[unit] = dp.unit_colors[unit]
        else:
            unit_ids = dp.unit_ids
            unit_colors = dp.unit_colors
        labels = dp.unit_ids

        for i, unit in enumerate(unit_ids):
            locs = spike_locations[unit]

            zorder = 5 if unit in dp.unit_ids else 3
            self.ax.scatter(locs["x"], locs["y"], s=2, alpha=0.3, color=unit_colors[unit], zorder=zorder)

        handles = [
            Line2D([0], [0], ls="", marker="o", markersize=5, markeredgewidth=2, color=unit_colors[unit])
            for unit in dp.unit_ids
        ]
        if dp.plot_legend:
            if hasattr(self, "legend") and self.legend is not None:
                self.legend.remove()
            self.legend = self.figure.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True, shadow=True
            )

        # set proper axis limits
        xlims, ylims = estimate_axis_lims(spike_locations)

        ax_xlims = list(self.ax.get_xlim())
        ax_ylims = list(self.ax.get_ylim())

        ax_xlims[0] = xlims[0] if xlims[0] < ax_xlims[0] else ax_xlims[0]
        ax_xlims[1] = xlims[1] if xlims[1] > ax_xlims[1] else ax_xlims[1]
        ax_ylims[0] = ylims[0] if ylims[0] < ax_ylims[0] else ax_ylims[0]
        ax_ylims[1] = ylims[1] if ylims[1] > ax_ylims[1] else ax_ylims[1]

        self.ax.set_xlim(ax_xlims)
        self.ax.set_ylim(ax_ylims)
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
        # TODO add an option checkbox for legend
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
        spike_locations = dp.spike_locations

        # ensure serializable for sortingview
        unit_ids, channel_ids = make_serializable(dp.unit_ids, dp.channel_ids)

        locations = {str(ch): dp.channel_locations[i_ch].astype("float32") for i_ch, ch in enumerate(channel_ids)}
        xlims, ylims = estimate_axis_lims(spike_locations)

        unit_items = []
        for unit in unit_ids:
            spike_times_sec = dp.sorting.get_unit_spike_train(
                unit_id=unit, segment_index=dp.segment_index, return_times=True
            )
            unit_items.append(
                vv.SpikeLocationsItem(
                    unit_id=unit,
                    spike_times_sec=spike_times_sec.astype("float32"),
                    x_locations=spike_locations[unit]["x"].astype("float32"),
                    y_locations=spike_locations[unit]["y"].astype("float32"),
                )
            )

        v_spike_locations = vv.SpikeLocations(
            units=unit_items,
            hide_unit_selector=dp.hide_unit_selector,
            x_range=xlims.astype("float32"),
            y_range=ylims.astype("float32"),
            channel_locations=locations,
            disable_auto_rotate=True,
        )

        if not dp.hide_unit_selector:
            v_units_table = generate_unit_table_view(dp.sorting)

            self.view = vv.Box(
                direction="horizontal",
                items=[
                    vv.LayoutItem(v_units_table, max_size=150),
                    vv.LayoutItem(v_spike_locations),
                ],
            )
        else:
            self.view = v_spike_locations

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)


def estimate_axis_lims(spike_locations, quantile=0.02):
    # set proper axis limits
    all_locs = np.concatenate(list(spike_locations.values()))
    xlims = np.quantile(all_locs["x"], [quantile, 1 - quantile])
    ylims = np.quantile(all_locs["y"], [quantile, 1 - quantile])

    return xlims, ylims
