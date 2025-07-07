from __future__ import annotations

import numpy as np
from warnings import warn

from .base import BaseWidget, default_backend_kwargs

from .amplitudes import AmplitudesWidget
from .crosscorrelograms import CrossCorrelogramsWidget
from .unit_templates import UnitTemplatesWidget

from .utils import get_some_colors

from spikeinterface.core.sortinganalyzer import SortingAnalyzer


class PotentialMergesWidget(BaseWidget):
    """
    Plots potential merges

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The input sorting analyzer
    potential_merges : list of lists or tuples
        List of potential merges (see `spikeinterface.curation.get_potential_auto_merges`)
    segment_index : int
        The segment index to display
    max_spike_samples : int or None, default: None
        The maximum number of spikes to display per unit
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        potential_merges: list,
        unit_colors: list = None,
        segment_index: int = 0,
        max_spikes_per_unit: int = 100,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        self.check_extensions(sorting_analyzer, ["templates", "spike_amplitudes", "correlograms"])

        unique_merge_units = np.unique([u for merge in potential_merges for u in merge])
        if unit_colors is None:
            unit_colors = get_some_colors(sorting_analyzer.unit_ids)

        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            potential_merges=potential_merges,
            unit_colors=unit_colors,
            segment_index=segment_index,
            max_spikes_per_unit=max_spikes_per_unit,
            unique_merge_units=unique_merge_units,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        from math import lcm
        import matplotlib.pyplot as plt

        # import ipywidgets.widgets as widgets
        import ipywidgets.widgets as W
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, ScaleWidget, WidenNarrowWidget

        check_ipywidget_backend()

        self.next_data_plot = data_plot.copy()

        cm = 1 / 2.54
        analyzer = data_plot["sorting_analyzer"]

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"] * 3

        ratios = [0.2, 0.8]

        with plt.ioff():
            output = W.Output()
            with output:
                self.figure = plt.figure(
                    figsize=((ratios[1] * width_cm) * cm, height_cm * cm),
                    constrained_layout=True,
                )
                plt.show()
        # find max number of merges:
        self.gs = None
        self.axes_amplitudes = None
        self.ax_templates = None
        self.ax_probe = None
        self.axes_cc = None

        # Instantiate sub-widgets
        self.w_amplitudes = AmplitudesWidget(
            analyzer,
            unit_colors=data_plot["unit_colors"],
            unit_ids=data_plot["unique_merge_units"],
            plot_histograms=True,
            plot_legend=False,
            immediate_plot=False,
        )
        self.w_templates = UnitTemplatesWidget(
            analyzer,
            unit_ids=data_plot["unique_merge_units"],
            unit_colors=data_plot["unit_colors"],
            plot_legend=False,
            immediate_plot=False,
        )
        self.w_crosscorrelograms = CrossCorrelogramsWidget(
            analyzer,
            unit_ids=data_plot["unique_merge_units"],
            min_similarity_for_correlograms=0,
            unit_colors=data_plot["unit_colors"],
            immediate_plot=False,
        )

        options = ["-".join([str(u) for u in m]) for m in data_plot["potential_merges"]]
        value = options[0]
        self.unit_selector_label = W.Label(value="Potential merges:")
        self.unit_selector = W.Dropdown(options=options, value=value, layout=W.Layout(width="80%"))
        self.previous_num_merges = len(data_plot["potential_merges"][0])
        self.scaler = ScaleWidget(value=1.0, layout=W.Layout(width="80%"))
        self.widen_narrow = WidenNarrowWidget(value=1.0, layout=W.Layout(width="80%"))

        left_sidebar = W.VBox(
            [self.unit_selector_label, self.unit_selector, self.scaler, self.widen_narrow],
            layout=W.Layout(width="100%"),
        )

        self.widget = W.AppLayout(
            center=self.figure.canvas,
            left_sidebar=left_sidebar,
            pane_widths=ratios + [0],
        )

        if len(np.unique([len(m) for m in self.data_plot["potential_merges"]])) == 1:
            # in this case we multiply the number of columns by 3 to have 2/3 of the space for the templates
            ncols = 3 * len(self.data_plot["potential_merges"][0])
        else:
            ncols = lcm(*[len(m) for m in self.data_plot["potential_merges"]])
        right_axes = int(ncols * 2 / 3)
        self.ncols = ncols
        self.right_axes = right_axes

        # a first update
        self._update_plot(None)

        self.unit_selector.observe(self._update_plot, names="value", type="change")
        self.scaler.observe(self._update_plot, names="value", type="change")
        self.widen_narrow.observe(self._update_plot, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _update_gs(self, merge_units):
        import matplotlib.gridspec as gridspec

        # we create a vertical grid with 1 row between the 3 first plots
        n_units = len(merge_units)
        ncols = self.ncols
        right_axes = self.right_axes
        unit_len_in_gs = self.ncols // n_units
        nrows = ncols * 3 + 2

        if self.gs is not None and self.previous_num_merges == len(merge_units):
            self.ax_templates.clear()
            self.ax_probe.clear()
            for ax in self.axes_amplitudes:
                ax.clear()
            for ax in self.axes_cc.flatten():
                ax.clear()
        else:
            self.figure.clear()
            self.gs = gridspec.GridSpec(nrows, ncols, figure=self.figure)
            self.ax_templates = self.figure.add_subplot(self.gs[:ncols, :right_axes])
            self.ax_probe = self.figure.add_subplot(self.gs[:ncols, right_axes:])
            row_offset = ncols + 1
            ax_amplitudes_ts = self.figure.add_subplot(self.gs[row_offset : row_offset + ncols, :right_axes])
            ax_amplitudes_hist = self.figure.add_subplot(self.gs[row_offset : row_offset + ncols, right_axes:])
            self.axes_amplitudes = [ax_amplitudes_ts, ax_amplitudes_hist]
            row_offset += ncols + 1
            self.axes_cc = []
            for i in range(0, n_units):
                for j in range(0, n_units):
                    self.axes_cc.append(
                        self.figure.add_subplot(
                            self.gs[
                                row_offset + (unit_len_in_gs) * i : row_offset + (unit_len_in_gs) * (i + 1),
                                j * unit_len_in_gs : (j + 1) * unit_len_in_gs,
                            ]
                        )
                    )
            self.axes_cc = np.array(self.axes_cc).reshape((n_units, n_units))
        self.previous_num_merges = len(merge_units)

    def _update_plot(self, change=None):

        merge_units = self.unit_selector.value
        sorting_analyzer = self.data_plot["sorting_analyzer"]
        channel_locations = sorting_analyzer.get_channel_locations()
        unit_ids = sorting_analyzer.unit_ids

        # unroll the merges
        unit_ids_str = [str(u) for u in unit_ids]
        plot_unit_ids = []
        for m in merge_units.split("-"):
            plot_unit_ids.append(unit_ids[unit_ids_str.index(m)])
        self._update_gs(plot_unit_ids)

        backend_kwargs_mpl = default_backend_kwargs["matplotlib"].copy()
        backend_kwargs_mpl.pop("axes")
        backend_kwargs_mpl.pop("ax")

        amplitude_data_plot = self.w_amplitudes.data_plot.copy()
        amplitude_data_plot["unit_ids"] = plot_unit_ids
        self.w_amplitudes.plot_matplotlib(amplitude_data_plot, ax=None, axes=self.axes_amplitudes, **backend_kwargs_mpl)

        unit_template_data_plot = self.w_templates.data_plot.copy()
        unit_template_data_plot["unit_ids"] = plot_unit_ids
        unit_template_data_plot["same_axis"] = True
        unit_template_data_plot["set_title"] = False
        unit_template_data_plot["scale"] = self.scaler.value
        unit_template_data_plot["widen_narrow_scale"] = self.widen_narrow.value
        # update templates and shading
        templates_ext = sorting_analyzer.get_extension("templates")
        unit_template_data_plot["templates"] = templates_ext.get_templates(unit_ids=plot_unit_ids, operator="average")
        unit_template_data_plot["templates_shading"] = self.w_templates._get_template_shadings(
            plot_unit_ids, self.w_templates.data_plot["templates_percentile_shading"]
        )
        self.w_templates.plot_matplotlib(unit_template_data_plot, ax=self.ax_templates, axes=None, **backend_kwargs_mpl)
        self.ax_templates.axis("off")
        self.w_templates._plot_probe(self.ax_probe, channel_locations, plot_unit_ids)
        crosscorrelograms_data_plot = self.w_crosscorrelograms.data_plot.copy()
        crosscorrelograms_data_plot["unit_ids"] = plot_unit_ids
        merge_unit_indices = np.flatnonzero(np.isin(self.data_plot["unique_merge_units"], plot_unit_ids))
        updated_correlograms = crosscorrelograms_data_plot["correlograms"]
        updated_correlograms = updated_correlograms[merge_unit_indices][:, merge_unit_indices]
        crosscorrelograms_data_plot["correlograms"] = updated_correlograms
        self.w_crosscorrelograms.plot_matplotlib(
            crosscorrelograms_data_plot, axes=self.axes_cc, ax=None, **backend_kwargs_mpl
        )
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
