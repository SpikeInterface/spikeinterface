import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget

from ..comparison import GroundTruthComparison


class SortingPerformanceWidget(BaseWidget):
    """
    Plots sorting performance for each ground-truth unit.

    Parameters
    ----------
    gt_sorting_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    property_name: str
        The property of the sorting extractor to use as x-axis (e.g. snr).
        If None, no property is used.
    metric: str
        The performance metric. 'accuracy' (default), 'precision', 'recall', 'miss rate', etc.
    markersize: int
        The size of the marker
    marker: str
        The matplotlib marker to use (default '.')
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: SortingPerformanceWidget
        The output widget
    """

    def __init__(self, sorting_comparison, metrics,
                 performance_name='accuracy', metric_name='snr',
                 markersize=10, marker='.', figure=None, ax=None):
        assert isinstance(sorting_comparison, GroundTruthComparison), \
            "The 'sorting_comparison' object should be a GroundTruthComparison instance"
        BaseWidget.__init__(self, figure, ax)
        self.sorting_comparison = sorting_comparison
        self.metrics = metrics
        self.performance_name = performance_name
        self.metric_name = metric_name
        self.markersize = markersize
        self.marker = marker

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        comp = self.sorting_comparison
        unit_ids = comp.sorting1.get_unit_ids()
        perf = comp.get_performance()[self.performance_name]
        metric = self.metrics[self.metric_name]

        ax = self.ax

        ax.plot(metric, perf, marker=self.marker, markersize=int(self.markersize), ls='')
        ax.set_xlabel(self.metric_name)
        ax.set_ylabel(self.performance_name)
        ax.set_ylim(0, 1.05)


def plot_sorting_performance(*args, **kwargs):
    W = SortingPerformanceWidget(*args, **kwargs)
    W.plot()
    return W


plot_sorting_performance.__doc__ = SortingPerformanceWidget.__doc__
