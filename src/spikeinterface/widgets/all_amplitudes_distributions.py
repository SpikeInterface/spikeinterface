from __future__ import annotations

import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_some_colors

from spikeinterface.core import SortingAnalyzer


class AllAmplitudesDistributionsWidget(BaseWidget):
    """
    Plots distributions of amplitudes as violin plot for all or some units.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer
    unit_ids : list
        List of unit ids, default None
    unit_colors : dict | None, default: None
        Dict of colors with unit ids as keys and colors as values. Colors can be any type accepted
        by matplotlib. If None, default colors are chosen using the `get_some_colors` function.
    """

    def __init__(
        self, sorting_analyzer: SortingAnalyzer, unit_ids=None, unit_colors=None, backend=None, **backend_kwargs
    ):

        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(sorting_analyzer, "spike_amplitudes")

        amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()

        num_segments = sorting_analyzer.get_num_segments()

        if unit_ids is None:
            unit_ids = sorting_analyzer.unit_ids

        if unit_colors is None:
            unit_colors = get_some_colors(sorting_analyzer.unit_ids)

        amplitudes_by_units = {}
        spikes = sorting_analyzer.sorting.to_spike_vector()
        for unit_id in unit_ids:
            unit_index = sorting_analyzer.sorting.id_to_index(unit_id)
            spike_mask = spikes["unit_index"] == unit_index
            amplitudes_by_units[unit_id] = amplitudes[spike_mask]

        plot_data = dict(
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            num_segments=num_segments,
            amplitudes_by_units=amplitudes_by_units,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        ax = self.ax

        parts = ax.violinplot(
            list(dp.amplitudes_by_units.values()), showmeans=False, showmedians=False, showextrema=False
        )

        for i, pc in enumerate(parts["bodies"]):
            color = dp.unit_colors[dp.unit_ids[i]]
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(1)

        ax.set_xticks(np.arange(len(dp.unit_ids)) + 1)
        ax.set_xticklabels([str(unit_id) for unit_id in dp.unit_ids])

        ylims = ax.get_ylim()
        if np.max(ylims) < 0:
            ax.set_ylim(min(ylims), 0)
        if np.min(ylims) > 0:
            ax.set_ylim(0, max(ylims))
