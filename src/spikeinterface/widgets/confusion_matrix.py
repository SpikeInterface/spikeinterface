import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_unit_colors


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
