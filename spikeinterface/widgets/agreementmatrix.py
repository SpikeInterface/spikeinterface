import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget


class AgreementMatrixWidget(BaseWidget):
    """
    Plots sorting comparison confusion matrix.

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
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created
    """

    def __init__(self, sorting_comparison, ordered=True, count_text=True, unit_ticks=True,
                 figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._sc = sorting_comparison
        self._ordered = ordered
        self._count_text = count_text
        self._unit_ticks = unit_ticks
        self.name = 'ConfusionMatrix'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        # a dataframe
        if self._ordered:
            scores = self._sc.get_ordered_agreement_scores()
        else:
            scores = self._sc.agreement_scores

        N1 = scores.shape[0]
        N2 = scores.shape[1]

        unit_ids1 = scores.index.values
        unit_ids2 = scores.columns.values

        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        self.ax.matshow(scores.values, cmap='Greens')

        if self._count_text:
            for i, u1 in enumerate(unit_ids1):
                u2 = self._sc.best_match_12[u1]
                if u2 != -1:
                    j = np.where(unit_ids2 == u2)[0][0]

                    self.ax.text(j, i, '{:0.2f}'.format(scores.at[u1, u2]),
                                 ha='center', va='center', color='white')

        # Major ticks
        self.ax.set_xticks(np.arange(0, N2))
        self.ax.set_yticks(np.arange(0, N1))
        self.ax.xaxis.tick_bottom()

        # Labels for major ticks
        if self._unit_ticks:
            self.ax.set_yticklabels(scores.index, fontsize=12)
            self.ax.set_xticklabels(scores.columns, fontsize=12)

        self.ax.set_xlabel(self._sc.name_list[1], fontsize=20)
        self.ax.set_ylabel(self._sc.name_list[0], fontsize=20)

        self.ax.set_xlim(-0.5, N2 - 0.5)
        self.ax.set_ylim(N1 - 0.5, -0.5, )


def plot_agreement_matrix(*args, **kwargs):
    W = AgreementMatrixWidget(*args, **kwargs)
    W.plot()
    return W


plot_agreement_matrix.__doc__ = AgreementMatrixWidget.__doc__
