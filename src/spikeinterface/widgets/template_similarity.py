from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr
from spikeinterface.core.sortinganalyzer import SortingAnalyzer


class TemplateSimilarityWidget(BaseWidget):
    """
    Plots unit template similarity.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The object to get template similarity from
    unit_ids : list or None, default: None
        List of unit ids default: None
    display_diagonal_values : bool, default: False
        If False, the diagonal is displayed as zeros.
        If True, the similarity values (all 1s) are displayed
    cmap : matplotlib colormap, default: "viridis"
        The matplotlib colormap
    show_unit_ticks : bool, default: False
        If True, ticks display unit ids
    show_colorbar : bool, default: True
        If True, color bar is displayed
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids=None,
        cmap="viridis",
        display_diagonal_values=False,
        show_unit_ticks=False,
        show_colorbar=True,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(sorting_analyzer, "template_similarity")

        tsc = sorting_analyzer.get_extension("template_similarity")
        similarity = tsc.get_data().copy()

        sorting = sorting_analyzer.sorting
        if unit_ids is None:
            unit_ids = sorting.unit_ids
        else:
            unit_indices = sorting.ids_to_indices(unit_ids)
            similarity = similarity[unit_indices][:, unit_indices]

        if not display_diagonal_values:
            np.fill_diagonal(similarity, 0)

        plot_data = dict(
            similarity=similarity,
            unit_ids=unit_ids,
            cmap=cmap,
            show_unit_ticks=show_unit_ticks,
            show_colorbar=show_colorbar,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        im = self.ax.matshow(dp.similarity, cmap=dp.cmap)

        if dp.show_unit_ticks:
            # Major ticks
            self.ax.set_xticks(np.arange(0, len(dp.unit_ids)))
            self.ax.set_yticks(np.arange(0, len(dp.unit_ids)))
            self.ax.xaxis.tick_bottom()

            # Labels for major ticks
            self.ax.set_yticklabels(dp.unit_ids, fontsize=12)
            self.ax.set_xticklabels(dp.unit_ids, fontsize=12)
        if dp.show_colorbar:
            self.figure.colorbar(im)

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import generate_unit_table_view, make_serializable, handle_display_and_url

        dp = to_attr(data_plot)

        # ensure serializable for sortingview
        unit_ids = make_serializable(dp.unit_ids)

        # similarity
        ss_items = []
        for i1, u1 in enumerate(unit_ids):
            for i2, u2 in enumerate(unit_ids):
                ss_items.append(
                    vv.UnitSimilarityScore(unit_id1=u1, unit_id2=u2, similarity=dp.similarity[i1, i2].astype("float32"))
                )

        self.view = vv.UnitSimilarityMatrix(unit_ids=list(unit_ids), similarity_scores=ss_items)

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)
