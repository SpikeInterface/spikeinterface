"""
Various widgets on top of GroundTruthStudy to summary results:
  * run times
  * performances
  * count units
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from .basewidget import BaseWidget



class StudyComparisonRunTimesWidget(BaseWidget):
    """
    Plot run times for a study.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    ax: matplotlib ax
        The ax to be used. If not given a figure is created
    color: 
        

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
    ax: matplotlib ax
        The ax to be used. If not given a figure is created
    cmap_name

    """
    def __init__(self, study, cmap_name='Set2',  ax=None):
        
        self.study = study
        self.cmap_name = cmap_name
        
        BaseWidget.__init__(self, ax=ax)
        
    def plot(self):
        study = self.study
        ax = self.ax
        
        count_units = study.aggregate_count_units()
        
        if study.exhaustive_gt:
            columns = ['num_well_detected', 'num_false_positive',  'num_redundant', 'num_overmerged']
        else:
            columns = ['num_well_detected', 'num_redundant', 'num_overmerged']
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
            x = np.arange(sorter_names.size) + 1 + c / (ncol + 2)
            if stds is None:
                yerr = None
            else:
                yerr = stds[col].values
            ax.bar(x, m[col].values, yerr=yerr, width=1/(ncol+2), color=cmap(c), label=clean_labels[c])

        ax.legend()

        ax.set_xticks(np.arange(sorter_names.size) + 1)
        ax.set_xticklabels(sorter_names, rotation=0, ha='left')
        
        ax.set_xlim(0, sorter_names.size + 1)
        
        if count_units['num_gt'].unique().size == 1:
            num_gt = count_units['num_gt'].unique()[0]
            ax.axhline(num_gt, ls='--', color='k')
        


def plot_gt_study_unit_counts(*args, **kwargs):
    W = StudyComparisonUnitCountWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_unit_counts.__doc__ = StudyComparisonUnitCountWidget.__doc__


class StudyComparisonPerformencesWidget(BaseWidget):
    """
    Plot run times for a study.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    ax: matplotlib ax
        The ax to be used. If not given a figure is created
    cmap_name

    """
    def __init__(self, study, palette='Set1',  ax=None):
        
        self.study = study
        self.palette = palette
        
        num_rec = len(study.rec_names)
        fig, axes = plt.subplots(ncols=1, nrows=num_rec, squeeze=False)
        
        BaseWidget.__init__(self, axes=axes)
        
    def plot(self):
        import seaborn as sns
        study = self.study
        
        
        sns.set_palette(sns.color_palette(self.palette))
        
        perf_by_units = study.aggregate_performance_by_unit()
        perf_by_units = perf_by_units.reset_index()
        
        for r, rec_name in enumerate(study.rec_names):
            ax = self.axes[r, 0]
            df = perf_by_units.loc[perf_by_units['rec_name'] == rec_name, :]
            df = pd.melt(df, id_vars='sorter_name', var_name='Metric', value_name='Score', 
                    value_vars=('accuracy','precision', 'recall'))
            sns.swarmplot(data=df, x='sorter_name', y='Score', hue='Metric', dodge=True,
                            s=3, ax=ax) # order=sorter_list,  
        #~ ax.set_xticklabels(sorter_names_short, rotation=30, ha='center')
        #~ ax.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0., frameon=False, fontsize=8, markerscale=0.5)        
            
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(f'Perfs for {rec_name}')

def plot_gt_study_performences(*args, **kwargs):
    W = StudyComparisonPerformencesWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_performences.__doc__ = StudyComparisonPerformencesWidget.__doc__

