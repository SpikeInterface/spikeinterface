from __future__ import annotations

import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_unit_colors


class ISIDistributionWidget(BaseWidget):
    """
    Plots spike train ISI distribution.

    Parameters
    ----------
    sorting : SortingExtractor
        The sorting extractor object
    unit_ids : list
        List of unit ids
    bins_ms : int
        Bin size in ms
    window_ms : float
        Window size in ms

    """

    def __init__(self, sorting, unit_ids=None, window_ms=100.0, bin_ms=1.0, backend=None, **backend_kwargs):
        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()

        plot_data = dict(
            sorting=sorting,
            unit_ids=unit_ids,
            window_ms=window_ms,
            bin_ms=bin_ms,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        if backend_kwargs.get("axes", None) is None:
            backend_kwargs["num_axes"] = len(dp.unit_ids)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        sorting = dp.sorting
        num_segments = sorting.get_num_segments()
        fs = sorting.sampling_frequency

        for i, unit_id in enumerate(dp.unit_ids):
            ax = self.axes.flatten()[i]

            bins = np.arange(0, dp.window_ms, dp.bin_ms)
            bin_counts = None
            for segment_index in range(num_segments):
                times_ms = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index) / fs * 1000.0
                isi = np.diff(times_ms)

                bin_counts_, bin_edges = np.histogram(isi, bins=bins, density=True)
                if segment_index == 0:
                    bin_counts = bin_counts_
                else:
                    bin_counts += bin_counts_
                    # TODO handle sensity when several segments

            ax.bar(x=bin_edges[:-1], height=bin_counts, width=dp.bin_ms, color="gray", align="edge")

            ax.set_ylabel(f"{unit_id}")
