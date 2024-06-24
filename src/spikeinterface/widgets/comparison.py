from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr


class ConfusionMatrixWidget(BaseWidget):
    """
    Plots sorting comparison confusion matrix.

    Parameters
    ----------
    gt_comparison : GroundTruthComparison
        The ground truth sorting comparison object
    count_text : bool
        If True counts are displayed as text
    unit_ticks : bool
        If True unit tick labels are displayed

    """

    def __init__(self, gt_comparison, count_text=True, unit_ticks=True, backend=None, **backend_kwargs):
        plot_data = dict(
            gt_comparison=gt_comparison,
            count_text=count_text,
            unit_ticks=unit_ticks,
        )
        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        comp = dp.gt_comparison

        confusion_matrix = comp.get_confusion_matrix()
        N1 = confusion_matrix.shape[0] - 1
        N2 = confusion_matrix.shape[1] - 1

        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        self.ax.matshow(confusion_matrix.values, cmap="Greens")

        if dp.count_text:
            for (i, j), z in np.ndenumerate(confusion_matrix.values):
                if z != 0:
                    if z > np.max(confusion_matrix.values) / 2.0:
                        self.ax.text(j, i, "{:d}".format(z), ha="center", va="center", color="white")
                    else:
                        self.ax.text(j, i, "{:d}".format(z), ha="center", va="center", color="black")

        self.ax.axhline(int(N1 - 1) + 0.5, color="black")
        self.ax.axvline(int(N2 - 1) + 0.5, color="black")

        # Major ticks
        self.ax.set_xticks(np.arange(0, N2 + 1))
        self.ax.set_yticks(np.arange(0, N1 + 1))
        self.ax.xaxis.tick_bottom()

        # Labels for major ticks
        if dp.unit_ticks:
            self.ax.set_yticklabels(confusion_matrix.index, fontsize=12)
            self.ax.set_xticklabels(confusion_matrix.columns, fontsize=12)
        else:
            self.ax.set_xticklabels(np.append([""] * N2, "FN"), fontsize=10)
            self.ax.set_yticklabels(np.append([""] * N1, "FP"), fontsize=10)

        self.ax.set_xlabel(comp.name_list[1], fontsize=20)
        self.ax.set_ylabel(comp.name_list[0], fontsize=20)

        self.ax.set_xlim(-0.5, N2 + 0.5)
        self.ax.set_ylim(
            N1 + 0.5,
            -0.5,
        )


class AgreementMatrixWidget(BaseWidget):
    """
    Plots sorting comparison agreement matrix.

    Parameters
    ----------
    sorting_comparison : GroundTruthComparison or SymmetricSortingComparison
        The sorting comparison object.
        Can optionally be symmetric if given a SymmetricSortingComparison
    ordered : bool, default: True
        Order units with best agreement scores.
        If True, agreement scores can be seen along a diagonal
    count_text : bool, default: True
        If True counts are displayed as text
    unit_ticks : bool, default: True
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
        self.ax.xaxis.tick_bottom()

        # Labels for major ticks
        if dp.unit_ticks:
            self.ax.set_xticks(np.arange(0, N2))
            self.ax.set_yticks(np.arange(0, N1))
            self.ax.set_yticklabels(scores.index)
            self.ax.set_xticklabels(scores.columns)

        self.ax.set_xlabel(comp.name_list[1])
        self.ax.set_ylabel(comp.name_list[0])

        self.ax.set_xlim(-0.5, N2 - 0.5)
        self.ax.set_ylim(
            N1 - 0.5,
            -0.5,
        )
