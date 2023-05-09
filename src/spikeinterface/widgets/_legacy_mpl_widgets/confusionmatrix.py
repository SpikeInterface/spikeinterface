import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget


class ConfusionMatrixWidget(BaseWidget):
    """
    Plots sorting comparison confusion matrix.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    count_text: bool
        If True counts are displayed as text
    unit_ticks: bool
        If True unit tick labels are displayed
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: ConfusionMatrixWidget
        The output widget
    """

    def __init__(self, gt_comparison, count_text=True, unit_ticks=True,
                 figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._gtcomp = gt_comparison
        self._count_text = count_text
        self._unit_ticks = unit_ticks
        self.name = 'ConfusionMatrix'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        # a dataframe
        confusion_matrix = self._gtcomp.get_confusion_matrix()

        N1 = confusion_matrix.shape[0] - 1
        N2 = confusion_matrix.shape[1] - 1

        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        self.ax.matshow(confusion_matrix.values, cmap='Greens')

        if self._count_text:
            for (i, j), z in np.ndenumerate(confusion_matrix.values):
                if z != 0:
                    if z > np.max(confusion_matrix.values) / 2.:
                        self.ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='white')
                    else:
                        self.ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='black')

        self.ax.axhline(int(N1 - 1) + 0.5, color='black')
        self.ax.axvline(int(N2 - 1) + 0.5, color='black')

        # Major ticks
        self.ax.set_xticks(np.arange(0, N2 + 1))
        self.ax.set_yticks(np.arange(0, N1 + 1))
        self.ax.xaxis.tick_bottom()

        # Labels for major ticks
        if self._unit_ticks:
            self.ax.set_yticklabels(confusion_matrix.index, fontsize=12)
            self.ax.set_xticklabels(confusion_matrix.columns, fontsize=12)
        else:
            self.ax.set_xticklabels(np.append([''] * N2, 'FN'), fontsize=10)
            self.ax.set_yticklabels(np.append([''] * N1, 'FP'), fontsize=10)

        self.ax.set_xlabel(self._gtcomp.name_list[1], fontsize=20)
        self.ax.set_ylabel(self._gtcomp.name_list[0], fontsize=20)

        self.ax.set_xlim(-0.5, N2 + 0.5)
        self.ax.set_ylim(N1 + 0.5, -0.5, )


def plot_confusion_matrix(*args, **kwargs):
    W = ConfusionMatrixWidget(*args, **kwargs)
    W.plot()
    return W


plot_confusion_matrix.__doc__ = ConfusionMatrixWidget.__doc__
