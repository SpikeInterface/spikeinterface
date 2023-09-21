import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_unit_colors


class AgreementMatrixWidget(BaseWidget):
    """
    Plots sorting comparison agreement matrix.

    Parameters
    ----------
    sorting_comparison: GroundTruthComparison or SymmetricSortingComparison
        The sorting comparison object.
        Symetric or not.
    ordered: bool
        Order units with best agreement scores.
        This enable to see agreement on a diagonal.
    count_text: bool
        If True counts are displayed as text
    unit_ticks: bool
        If True unit tick labels are displayed

    """

    def __init__(
        self, sorting_comparison, ordered=True, count_text=True, unit_ticks=True, backend=None, **backend_kwargs
    ):
        plot_data = dict(
            sorting_comparison=sorting_comparison,
            ordered=ordered,
            count_text=count_text,
            unit_ticks=unit_ticks,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        comp = dp.sorting_comparison

        if dp.ordered:
            scores = comp.get_ordered_agreement_scores()
        else:
            scores = comp.agreement_scores

        N1 = scores.shape[0]
        N2 = scores.shape[1]

        unit_ids1 = scores.index.values
        unit_ids2 = scores.columns.values

        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        self.ax.matshow(scores.values, cmap="Greens")

        if dp.count_text:
            for i, u1 in enumerate(unit_ids1):
                u2 = comp.best_match_12[u1]
                if u2 != -1:
                    j = np.where(unit_ids2 == u2)[0][0]

                    self.ax.text(j, i, "{:0.2f}".format(scores.at[u1, u2]), ha="center", va="center", color="white")

        # Major ticks
        self.ax.set_xticks(np.arange(0, N2))
        self.ax.set_yticks(np.arange(0, N1))
        self.ax.xaxis.tick_bottom()

        # Labels for major ticks
        if dp.unit_ticks:
            self.ax.set_yticklabels(scores.index, fontsize=12)
            self.ax.set_xticklabels(scores.columns, fontsize=12)

        self.ax.set_xlabel(comp.name_list[1], fontsize=20)
        self.ax.set_ylabel(comp.name_list[0], fontsize=20)

        self.ax.set_xlim(-0.5, N2 - 0.5)
        self.ax.set_ylim(
            N1 - 0.5,
            -0.5,
        )
