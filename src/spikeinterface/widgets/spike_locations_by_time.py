from __future__ import annotations

import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_some_colors

from spikeinterface.core.sortinganalyzer import SortingAnalyzer


class LocationsWidget(BaseWidget):
    """
    Plots spike locations as a function of time

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The input sorting analyzer
    unit_ids : list or None, default: None
        List of unit ids
    segment_index : int or None, default: None
        The segment index (or None if mono-segment)
    max_spikes_per_unit : int or None, default: None
        Number of max spikes per unit to display. Use None for all spikes
    plot_histogram : bool, default: False
        If True, an histogram of the locations is plotted on the right axis
        (matplotlib backend)
    bins : int or None, default: None
        If plot_histogram is True, the number of bins for the location histogram.
        If None this is automatically adjusted
    plot_legend : bool, default: True
        True includes legend in plot
    locations_axis : str, default: 'y'
        Which location axis to use when plotting locations.
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids=None,
        unit_colors=None,
        segment_index=None,
        max_spikes_per_unit=None,
        plot_histograms=False,
        bins=None,
        plot_legend=True,
        locations_axis="y",
        backend=None,
        **backend_kwargs,
    ):

        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        sorting = sorting_analyzer.sorting
        self.check_extensions(sorting_analyzer, "spike_locations")

        locations = sorting_analyzer.get_extension("spike_locations").get_data(outputs="by_unit")

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        if unit_colors is None:
            unit_colors = get_some_colors(sorting.unit_ids)

        if sorting.get_num_segments() > 1:
            if segment_index is None:
                warn("More than one segment available! Using segment_index 0")
                segment_index = 0
        else:
            segment_index = 0
        locations_segment = locations[segment_index]
        total_duration = sorting_analyzer.get_num_samples(segment_index) / sorting_analyzer.sampling_frequency

        spiketrains_segment = {}
        for i, unit_id in enumerate(sorting.unit_ids):
            times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
            times = times / sorting.get_sampling_frequency()
            spiketrains_segment[unit_id] = times

        all_spiketrains = spiketrains_segment
        all_locations = locations_segment
        if max_spikes_per_unit is not None:
            spiketrains_to_plot = dict()
            locations_to_plot = dict()
            for unit, st in all_spiketrains.items():
                locs = all_locations[unit][locations_axis]
                if len(st) > max_spikes_per_unit:
                    random_idxs = np.random.choice(len(st), size=max_spikes_per_unit, replace=False)
                    spiketrains_to_plot[unit] = st[random_idxs]
                    locations_to_plot[unit] = locs[random_idxs]
                else:
                    spiketrains_to_plot[unit] = st
                    locations_to_plot[unit] = locs
        else:
            spiketrains_to_plot = all_spiketrains
            locations_to_plot = {
                unit_id: all_locations[unit_id][locations_axis] for unit_id in sorting_analyzer.unit_ids
            }

        if plot_histograms and bins is None:
            bins = 100

        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            locations=locations_to_plot,
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            spiketrains=spiketrains_to_plot,
            total_duration=total_duration,
            plot_histograms=plot_histograms,
            bins=bins,
            plot_legend=plot_legend,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        if backend_kwargs["axes"] is not None:
            axes = backend_kwargs["axes"]
            if dp.plot_histograms:
                assert np.asarray(axes).size == 2
            else:
                assert np.asarray(axes).size == 1
        elif backend_kwargs["ax"] is not None:
            assert not dp.plot_histograms
        else:
            if dp.plot_histograms:
                backend_kwargs["num_axes"] = 2
                backend_kwargs["ncols"] = 2
            else:
                backend_kwargs["num_axes"] = None

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        scatter_ax = self.axes.flatten()[0]

        for unit_id in dp.unit_ids:
            spiketrains = dp.spiketrains[unit_id]
            locs = dp.locations[unit_id]
            scatter_ax.scatter(spiketrains, locs, color=dp.unit_colors[unit_id], s=3, alpha=1, label=unit_id)

            if dp.plot_histograms:
                if dp.bins is None:
                    bins = int(len(spiketrains) / 30)
                else:
                    bins = dp.bins
                ax_hist = self.axes.flatten()[1]
                count, bins = np.histogram(locs, bins=bins)
                ax_hist.plot(count, bins[:-1], color=dp.unit_colors[unit_id], alpha=0.8)

        if dp.plot_histograms:
            ax_hist = self.axes.flatten()[1]
            ax_hist.set_ylim(scatter_ax.get_ylim())
            ax_hist.axis("off")
            # self.figure.tight_layout()

        if dp.plot_legend:
            if hasattr(self, "legend") and self.legend is not None:
                self.legend.remove()
            self.legend = self.figure.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True, shadow=True
            )

        scatter_ax.set_xlim(0, dp.total_duration)
        scatter_ax.set_xlabel("Times [s]")
        scatter_ax.set_ylabel(f"Location [um]")
        scatter_ax.spines["top"].set_visible(False)
        scatter_ax.spines["right"].set_visible(False)
        self.figure.subplots_adjust(bottom=0.1, top=0.9, left=0.1)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt

        # import ipywidgets.widgets as widgets
        import ipywidgets.widgets as W
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()

        cm = 1 / 2.54
        analyzer = data_plot["sorting_analyzer"]

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = W.Output()
            with output:
                self.figure = plt.figure(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        self.unit_selector = UnitSelector(analyzer.unit_ids)
        self.unit_selector.value = list(analyzer.unit_ids)[:1]

        self.checkbox_histograms = W.Checkbox(
            value=data_plot["plot_histograms"],
            description="hist",
        )

        left_sidebar = W.VBox(
            children=[
                self.unit_selector,
                self.checkbox_histograms,
            ],
            layout=W.Layout(align_items="center", width="100%", height="100%"),
        )

        self.widget = W.AppLayout(
            center=self.figure.canvas,
            left_sidebar=left_sidebar,
            pane_widths=ratios + [0],
        )

        # a first update
        self._full_update_plot()

        self.unit_selector.observe(self._update_plot, names="value", type="change")
        self.checkbox_histograms.observe(self._full_update_plot, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _full_update_plot(self, change=None):
        self.figure.clear()
        data_plot = self.next_data_plot
        data_plot["unit_ids"] = self.unit_selector.value
        data_plot["plot_histograms"] = self.checkbox_histograms.value
        data_plot["plot_legend"] = False

        backend_kwargs = dict(figure=self.figure, axes=None, ax=None)
        self.plot_matplotlib(data_plot, **backend_kwargs)
        self._update_plot()

    def _update_plot(self, change=None):
        for ax in self.axes.flatten():
            ax.clear()

        data_plot = self.next_data_plot
        data_plot["unit_ids"] = self.unit_selector.value
        data_plot["plot_histograms"] = self.checkbox_histograms.value
        data_plot["plot_legend"] = False

        backend_kwargs = dict(figure=None, axes=self.axes, ax=None)
        self.plot_matplotlib(data_plot, **backend_kwargs)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
