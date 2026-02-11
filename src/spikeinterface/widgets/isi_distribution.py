from __future__ import annotations

import numpy as np
from warnings import warn

from spikeinterface.core import SortingAnalyzer, BaseSorting

from .base import BaseWidget, to_attr


class ISIDistributionWidget(BaseWidget):
    """
    Plots spike train ISI distribution.

    Parameters
    ----------
    sorting_analyzer_or_sorting : SortingAnalyzer | BaseSorting | None, default: None
        The object containing the sorting information for the isi distribution plot
    unit_ids : list | None, default: None
        List of unit ids. If None, uses all unit ids.
    window_ms : float, default: 100.0
        Window size in ms
    bins_ms : int, default: 1.0
        Bin size in ms
    sorting : SortingExtractor | None, default: None
        A sorting object. Deprecated.
    """

    def __init__(
        self,
        sorting_analyzer_or_sorting: SortingAnalyzer | BaseSorting | None = None,
        unit_ids: list | None = None,
        window_ms: float = 100.0,
        bin_ms: float = 1.0,
        backend: str | None = None,
        sorting: BaseSorting | None = None,
        **backend_kwargs,
    ):

        if sorting is not None:
            # When removed, make `sorting_analyzer_or_sorting` a required argument rather than None.
            deprecation_msg = "`sorting` argument is deprecated and will be removed in version 0.105.0. Please use `sorting_analyzer_or_sorting` instead"
            warn(deprecation_msg, category=DeprecationWarning, stacklevel=2)
            sorting_analyzer_or_sorting = sorting

        sorting = self.ensure_sorting(sorting_analyzer_or_sorting)

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
