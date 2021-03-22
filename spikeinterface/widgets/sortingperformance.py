from spikewidgets.widgets.basewidget import BaseWidget
import spikecomparison as sc


def plot_sorting_performance(gt_sorting_comparison, property_name=None, metric='accuracy', markersize=10, marker='.',
                             figure=None, ax=None):
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
    W = SortingPerformanceWidget(
        sorting_comparison=gt_sorting_comparison,
        property_name=property_name,
        figure=figure,
        markersize=markersize,
        marker=marker,
        metric=metric,
        ax=ax
    )
    W.plot()
    return W


class SortingPerformanceWidget(BaseWidget):
    def __init__(self, *, sorting_comparison, property_name=None, metric='accuracy', markersize=10,
                 marker='.', figure=None, ax=None):
        assert isinstance(sorting_comparison, sc.GroundTruthComparison), \
            "The 'sorting_comparison' object should be a GroundTruthComparison instance"
        BaseWidget.__init__(self, figure, ax)
        self._SC = sorting_comparison
        self._property_name = property_name
        self._metric = metric
        self._ms = markersize
        self._mark = marker
        self.name = 'SortingPerformance'

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        SC = self._SC
        units = SC.sorting1.get_unit_ids()
        perf = SC.get_performance()[self._metric]
        if self._property_name is not None:
            assert self._property_name in SC.sorting1.get_shared_unit_property_names(), "%s should be " \
                                                                                 "a property of the ground truth " \
                                                                                 "sorting extractor"
            xvals = SC.sorting1.get_units_property(unit_ids=units, property_name=self._property_name)
            self.ax.plot(xvals, perf, marker=self._mark, markersize=int(self._ms), ls='')
            self.ax.set_xlabel(self._property_name)
        else:
            self.ax.plot(perf, '.')
            self.ax.set_xticks([])
        self.ax.set_ylabel(self._metric)
        self.ax.set_ylim([-0.05, 1.05])
