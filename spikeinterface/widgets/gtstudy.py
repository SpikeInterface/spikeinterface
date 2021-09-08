"""
Various widgets on top of GroundTruthStudy to summary results:
  * run times
  * performances
  * count units
"""
import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget



class StudyComparisonRunTimesWidget(BaseWidget):
    """
    Plot run times for a study.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    figure: matplotlib figure
        The figure to be used. If not given a figure is created

    Returns
    -------
    W: ConfusionMatrixWidget
        The output widget
    """
    def __init__(self, study, color='#F7DC6F',  ax=None):
        
        self.study = study
        self.color = color
        
        BaseWidget.__init__(self, ax=ax)
        
    def plot(self):
        study = self.study
        ax = self.ax
        
        
        
        all_run_times = study.aggregate_run_times()
        av_run_times = all_run_times.reset_index().groupby('sorter_name')['run_time'].mean()
        
        if len(study.rec_names) == 1:
            # no errors bars
            yerr = None
        else:
            # errors bars across recording
            yerr = all_run_times.reset_index().groupby('sorter_name')['run_time'].std()
        
        sorter_names = av_run_times.index
        
        x = np.arange(sorter_names.size) + 1
        ax.bar(x, av_run_times.values, width= 0.8, color=self.color, yerr=yerr)
        ax.set_ylabel('run times [s]')
        ax.set_xticks(x)
        ax.set_xticklabels(sorter_names)
        ax.set_xlim(0, sorter_names.size + 1)



def plot_gt_study_run_times(*args, **kwargs):
    W = StudyComparisonRunTimesWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_run_times.__doc__ = StudyComparisonRunTimesWidget.__doc__


class StudyComparisonUnitCountWidget(BaseWidget):
    """
    Plot run times for a study.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    figure: matplotlib figure
        The figure to be used. If not given a figure is created

    Returns
    -------
    W: ConfusionMatrixWidget
        The output widget
    """
    def __init__(self, study, exhaustive_gt=False, cmap_name='Set2',  ax=None):
        
        self.study = study
        self.cmap_name = cmap_name
        
        BaseWidget.__init__(self, ax=ax)
        
    def plot(self):
        study = self.study
        ax = self.ax
        
        count_units = study.aggregate_count_units()
        
        columns = ['num_well_detected', 'num_redundant', 'num_overmerged']
        if study.exhaustive_gt:
            columns += ['num_false_positive']
        ncol = len(columns)
            
        df = count_units.reset_index()
        
        m = df.groupby('sorter_name')[columns].mean()
        
        cmap = plt.get_cmap(self.cmap_name, 4)
        
        if len(study.rec_names) == 1:
            # no errors bars
            stds = None
        else:
            # errors bars across recording
            stds = df.groupby('sorter_name')[columns].std()
        
        sorter_names = m.index
        clean_labels = [col.replace('num_', '').replace('_', ' ').title() for col in columns]
        
        for c, col in enumerate(columns):
            x = np.arange(sorter_names.size) + 1 + c / (ncol + 1)
            if stds is None:
                yerr = None
            else:
                yerr = stds[col].values
            ax.bar(x, m[col].values, yerr=yerr, width=1/(ncol+1), color=cmap(c), label=clean_labels[c])

        ax.legend()

        ax.set_xticks(np.arange(sorter_names.size) + 1.4)
        ax.set_xticklabels(sorter_names)
        
        ax.set_xlim(0, sorter_names.size + 1)




def plot_gt_study_unit_counts(*args, **kwargs):
    W = StudyComparisonUnitCountWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_unit_counts.__doc__ = StudyComparisonUnitCountWidget.__doc__


