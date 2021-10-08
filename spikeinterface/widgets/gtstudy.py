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
        ax.bar(x, av_run_times.values, width=0.8, color=self.color, yerr=yerr)
        ax.set_ylabel('run times (s)')
        ax.set_xticks(x)
        ax.set_xticklabels(sorter_names, rotation=45)
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
        ax.set_xticklabels(sorter_names, rotation=45, ha='left')        
        ax.set_ylabel('# units')
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
        
    def plot(self, average=False):
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




class StudyComparisonTemplateSimilarityWidget(BaseWidget):
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
    def __init__(self, study, cmap_name='Set1',  ax=None):
        
        self.study = study
        self.cmap_name = cmap_name

        BaseWidget.__init__(self, ax=ax)
        
    def plot(self):
        import seaborn as sns
        study = self.study
        ax = self.ax

        perf_by_units = study.aggregate_performance_by_unit()
        perf_by_units = perf_by_units.reset_index()

        columns = ['accuracy','precision', 'recall']
        to_agg = {}
        ncol = len(columns)
        
        for column in columns:
            perf_by_units[column] = pd.to_numeric(perf_by_units[column], downcast='float')
            to_agg[column] = ['mean']

        data = perf_by_units.groupby(['sorter_name', 'rec_name']).agg(to_agg)
        
        m = data.groupby('sorter_name').mean()

        cmap = plt.get_cmap(self.cmap_name, 4)
        
        if len(study.rec_names) == 1:
            # no errors bars
            stds = None
        else:
            # errors bars across recording
            stds = data.groupby('sorter_name').std()
        
        sorter_names = m.index
        clean_labels = [col.replace('num_', '').replace('_', ' ').title() for col in columns]
        
        width = 1/(ncol+2)

        for c, col in enumerate(columns):
            x = np.arange(sorter_names.size) + 1 + c / (ncol + 2)
            if stds is None:
                yerr = None
            else:
                yerr = stds[col].values
            ax.bar(x, m[col].values.flatten(), yerr=yerr.flatten(), width=width, color=cmap(c), label=clean_labels[c])

        ax.legend()

        ax.set_xticks(np.arange(sorter_names.size) + 1 + width)
        ax.set_xticklabels(sorter_names, rotation=45)        
        ax.set_ylabel('metric')
        ax.set_xlim(0, sorter_names.size + 1)


class StudyComparisonPerformencesAveragesWidget(BaseWidget):
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
    def __init__(self, study, cmap_name='Set1',  ax=None):
        
        self.study = study
        self.cmap_name = cmap_name

        BaseWidget.__init__(self, ax=ax)
        
    def plot(self):
        import seaborn as sns
        study = self.study
        ax = self.ax

        perf_by_units = study.aggregate_performance_by_unit()
        perf_by_units = perf_by_units.reset_index()

        columns = ['accuracy','precision', 'recall']
        to_agg = {}
        ncol = len(columns)
        
        for column in columns:
            perf_by_units[column] = pd.to_numeric(perf_by_units[column], downcast='float')
            to_agg[column] = ['mean']

        data = perf_by_units.groupby(['sorter_name', 'rec_name']).agg(to_agg)
        
        m = data.groupby('sorter_name').mean()

        cmap = plt.get_cmap(self.cmap_name, 4)
        
        if len(study.rec_names) == 1:
            # no errors bars
            stds = None
        else:
            # errors bars across recording
            stds = data.groupby('sorter_name').std()
        
        sorter_names = m.index
        clean_labels = [col.replace('num_', '').replace('_', ' ').title() for col in columns]
        
        width = 1/(ncol+2)

        for c, col in enumerate(columns):
            x = np.arange(sorter_names.size) + 1 + c / (ncol + 2)
            if stds is None:
                yerr = None
            else:
                yerr = stds[col].values
            ax.bar(x, m[col].values.flatten(), yerr=yerr.flatten(), width=width, color=cmap(c), label=clean_labels[c])

        ax.legend()

        ax.set_xticks(np.arange(sorter_names.size) + 1 + width)
        ax.set_xticklabels(sorter_names, rotation=45)        
        ax.set_ylabel('metric')
        ax.set_xlim(0, sorter_names.size + 1)
        


class StudyComparisonPerformencesByTemplateSimilarity(BaseWidget):
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
    def __init__(self, study, cmap_name='Set1',  ax=None, ylim=(0.6, 1), show_legend=True):
        
        self.study = study
        self.cmap_name = cmap_name
        self.show_legend = show_legend
        self.ylim = ylim

        BaseWidget.__init__(self, ax=ax)
        
    def plot(self):

        import sklearn

        cmap = plt.get_cmap(self.cmap_name, len(self.study.sorter_names))
        colors = [cmap(i) for i in range(len(self.study.sorter_names))]
                
        flat_templates_gt = {}
        for rec_name in self.study.rec_names:

            waveform_folder = self.study.study_folder / 'waveforms' / f'waveforms_GroundTruth_{rec_name}'
            if not waveform_folder.is_dir():
                self.study.compute_waveforms(rec_name)

            templates = self.study.get_templates(rec_name)
            flat_templates_gt[rec_name] = templates.reshape(templates.shape[0], -1)
        
        all_results = {}

        for sorter_name in self.study.sorter_names:

            all_results[sorter_name] = {'similarity' : [], 'accuracy' : []}

            for rec_name in self.study.rec_names:

                try:
                    waveform_folder = self.study.study_folder / 'waveforms' / f'waveforms_{sorter_name}_{rec_name}'
                    if not waveform_folder.is_dir():
                        self.study.compute_waveforms(rec_name, sorter_name)
                    templates = self.study.get_templates(rec_name, sorter_name)
                    flat_templates = templates.reshape(templates.shape[0], -1)
                    similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates_gt[rec_name], flat_templates)

                    comp = self.study.comparisons[(rec_name, sorter_name)]

                    for i, u1 in enumerate(comp.sorting1.unit_ids):
                        u2 = comp.best_match_12[u1]
                        if u2 != -1:
                            all_results[sorter_name]['similarity'] += [similarity_matrix[comp.sorting1.id_to_index(u1), comp.sorting2.id_to_index(u2)]]
                            all_results[sorter_name]['accuracy'] += [comp.agreement_scores.at[u1, u2]]
                except Exception:
                    pass

            all_results[sorter_name]['similarity'] = np.array(all_results[sorter_name]['similarity'])
            all_results[sorter_name]['accuracy'] = np.array(all_results[sorter_name]['accuracy'])

        from matplotlib.patches import Ellipse

        similarity_means = [all_results[sorter_name]['similarity'].mean() for sorter_name in self.study.sorter_names]
        similarity_stds = [all_results[sorter_name]['similarity'].std() for sorter_name in self.study.sorter_names]

        accuracy_means = [all_results[sorter_name]['accuracy'].mean() for sorter_name in self.study.sorter_names]
        accuracy_stds = [all_results[sorter_name]['accuracy'].std() for sorter_name in self.study.sorter_names]

        scount = 0
        for x,y, i,j in zip(similarity_means, accuracy_means, similarity_stds, accuracy_stds):
            e = Ellipse((x,y), i, j)
            e.set_alpha(0.2)
            e.set_facecolor(colors[scount])
            self.ax.add_artist(e)
            self.ax.scatter([x], [y], c=colors[scount], label=self.study.sorter_names[scount])
            scount += 1

        self.ax.set_ylabel('accuracy')
        self.ax.set_xlabel('cosine similarity')
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

        if self.show_legend:
            self.ax.legend()



def plot_gt_study_performences(*args, **kwargs):
    W = StudyComparisonPerformencesWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_performences.__doc__ = StudyComparisonPerformencesWidget.__doc__

def plot_gt_study_performences_averages(*args, **kwargs):
    W = StudyComparisonPerformencesAveragesWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_performences_averages.__doc__ = StudyComparisonPerformencesAveragesWidget.__doc__


def plot_gt_study_performences_by_template_similarity(*args, **kwargs):
    W = StudyComparisonPerformencesByTemplateSimilarity(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_performences_by_template_similarity.__doc__ = StudyComparisonPerformencesByTemplateSimilarity.__doc__
