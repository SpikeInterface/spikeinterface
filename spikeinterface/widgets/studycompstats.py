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
    def __init__(self, study, exhaustive_gt=False, figure=None, ax=None, axes=None):

        if exhaustive_gt:
            self._ncols = 3
        else:
            self.n_cols = 2

        self._nrows = 3

        if axes is None and ax is None:
            figure, axes = plt.subplots(nrows=self._nrows, ncols=self._ncols)
        
        BaseWidget.__init__(self, figure, ax, axes)
        self._study = study
        self.sorter_names = self._study.sorter_names
        self.rec_names = study.rec_names
        self._exhaustive_gt = exhaustive_gt
        self.name = 'StudyComparisonStatistics'
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

    def get_all(self):
        res = self.run_times()
        for sorter in self.sorter_names:
            res[sorter].update(self.performance_by_units()[sorter])
            res[sorter].update(self.count_units()[sorter])
        return res

    def get_metrics(self, **kwargs):
        res = {}
        for rec in self._study.rec_names:
            res[rec] = self._study.get_metrics(rec, **kwargs)
        return res

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        
        data = self.get_all()
        if self._exhaustive_gt:
            to_display = ['run_time', 'num_well_detected', 'num_redundant', 'num_overmerged',
                        'num_bad', 'num_false_positive', 'accuracy', 'precision', 'recall']
        else:
            to_display = ['run_time', 'num_well_detected', 'num_redundant', 'num_overmerged',
                        'accuracy']

        nb_sorters = len(self.sorter_names)

        colors = ['C%d' %i for i in range(nb_sorters)]

        for scount, key in enumerate(to_display):

            ax = self.get_tiled_ax(scount, self._nrows, self._ncols)

            means = [data[sorter][key]['mean'] for sorter in self.sorter_names]
            stds = [data[sorter][key]['std'] for sorter in self.sorter_names]
            ax.bar(np.arange(nb_sorters), np.nan_to_num(means), yerr=np.nan_to_num(stds), color=colors)
            ax.set_ylabel(key)

            if (scount // self._ncols) == (self._nrows - 1):
                ax.set_xticks(np.arange(nb_sorters))
                ax.set_xticklabels(self.sorter_names, rotation=45)
            else:
                ax.tick_params(labelbottom=False)
