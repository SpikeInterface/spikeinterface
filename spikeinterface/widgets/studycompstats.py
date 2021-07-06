import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget
from spikeinterface.toolkit import compute_correlograms



class StudyComparisonStatisticsWidget(BaseWidget):
    """
    Plots difference between real CC matrix, and reconstructed one

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    in_ms:  float
        bins duration in ms
    window_ms: float
        Window duration in ms
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: ConfusionMatrixWidget
        The output widget
    """
    def __init__(self, study, exhaustive_gt=False, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        self._study = study
        self.sorter_names = study.sorter_names
        self.rec_names = study.rec_names
        self._exhaustive_gt = exhaustive_gt
        self.name = 'StudyComparisonStatistics'
        self._sorters = study.sorter_names
        self._compute()

    def _compute(self):
        self._study.run_comparisons(exhaustive_gt=self._exhaustive_gt)

    def count_units(self):
        res = {}    
        data = self._study.aggregate_count_units()
        keys = data.keys()

        for sorter in self.sorter_names:
            res[sorter] = {}
            for key in keys:
                res[sorter][key] = {}
                res[sorter][key]['mean'] = data.loc[:, sorter, :][key].mean()
                res[sorter][key]['std'] = data.loc[:, sorter, :][key].std()
            return res

    def performance_by_units(self):
        res = {}
        data = self._study.aggregate_performance_by_units()
        keys = data.keys()
        for sorter in self.sorter_names:
            res[sorter] = {}
            for key in keys:
                res[sorter][key] = {}
                res[sorter][key]['mean'] = data.loc[:, sorter, :][key].mean()
                res[sorter][key]['std'] = data.loc[:, sorter, :][key].std()
            return res

    def run_times(self):
        res = {}
        data = self._study.aggregate_run_times()
        keys = data.keys()
        for sorter in self.sorter_names:
            res[sorter] = {}
            for key in keys:
                res[sorter][key] = {}
                res[sorter][key]['mean'] = data.loc[:, sorter, :][key].mean()
                res[sorter][key]['std'] = data.loc[:, sorter, :][key].std()
            return res

    def get_metrics(self, **kwargs):
        res = {}
        for rec in self._study.rec_names:
            res[rec] = self._study.get_metrics(rec, **kwargs)
        return res

    def plot(self):
        self._do_plot()

    def _do_plot(self):

        fig = self.figure
        