from __future__ import annotations

import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_some_colors

from ..core import SortingResult


class AllAmplitudesDistributionsWidget(BaseWidget):
    """
    Plots distributions of amplitudes as violin plot for all or some units.

    Parameters
    ----------
    sorting_result: SortingResult
        The SortingResult
    unit_ids: list
        List of unit ids, default None
    unit_colors: None or dict
        Dict of colors with key: unit, value: color, default None
    """

    def __init__(
        self, sorting_result: SortingResult, unit_ids=None, unit_colors=None, backend=None, **backend_kwargs
    ):

        self.check_extensions(sorting_result, "spike_amplitudes")
        
        amplitudes = sorting_result.get_extension("spike_amplitudes").get_data()

        num_segments = sorting_result.get_num_segments()

        if unit_ids is None:
            unit_ids = sorting_result.unit_ids

        if unit_colors is None:
            unit_colors = get_some_colors(sorting_result.unit_ids)

        plot_data = dict(
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            num_segments=num_segments,
            amplitudes=amplitudes,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        ax = self.ax

        unit_amps = []
        for i, unit_id in enumerate(dp.unit_ids):
            amps = []
            for segment_index in range(dp.num_segments):
                amps.append(dp.amplitudes[segment_index][unit_id])
            amps = np.concatenate(amps)
            unit_amps.append(amps)
        parts = ax.violinplot(unit_amps, showmeans=False, showmedians=False, showextrema=False)

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
