"""
Various widgets on top of GroundTruthStudy to summary results:
  * run times
  * performancess
  * count units
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from .basewidget import BaseWidget


class ComparisonPerformancesWidget(BaseWidget):
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
    def __init__(self, gt_comp, palette='Set1',  ax=None):

        self.gt_comp = gt_comp
        self.palette = palette

        BaseWidget.__init__(self, ax=ax)

    def plot(self):
        import seaborn as sns
        ax = self.ax

        sns.set_palette(sns.color_palette(self.palette))

        perf_by_units = self.gt_comp.get_performance()
        perf_by_units = perf_by_units.reset_index()

        df = pd.melt(perf_by_units, var_name='Metric', value_name='Score',
                    value_vars=('accuracy','precision', 'recall'))
        import seaborn as sns
        sns.swarmplot(data=df, x="Metric", y='Score', hue='Metric', dodge=True,
                                    s=3, ax=ax) # order=sorter_list,
        #~ ax.set_xticklabels(sorter_names_short, rotation=30, ha='center')
        #~ ax.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0., frameon=False, fontsize=8, markerscale=0.5)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f'Performance')



class ComparisonPerformancesAveragesWidget(BaseWidget):
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
    def __init__(self, gt_comp, cmap_name='Set1',  ax=None):

        self.gt_comp = gt_comp
        self.cmap_name = cmap_name

        BaseWidget.__init__(self, ax=ax)

    def plot(self):
        import seaborn as sns
        ax = self.ax

        perf_by_units = self.gt_comp.get_performance()
        perf_by_units = perf_by_units.reset_index()

        columns = ['accuracy','precision', 'recall']
        to_agg = {}
        ncol = len(columns)

        for column in columns:
            perf_by_units[column] = pd.to_numeric(perf_by_units[column], downcast='float')
            to_agg[column] = ['mean', 'std']

        data = perf_by_units.agg(to_agg)

        m = data.mean()

        cmap = plt.get_cmap(self.cmap_name, 4)

        stds = data.std()

        clean_labels = [col.replace('num_', '').replace('_', ' ').title() for col in columns]

        width = 1/(ncol+2)

        for c, col in enumerate(columns):
            x = 1 + c / (ncol + 2)
            yerr = stds[col]
            ax.bar(x, m[col], yerr=yerr, width=width, color=cmap(c), label=clean_labels[c])

        ax.legend()
        ax.set_ylabel('metric')
        #ax.set_xlim(0, 1)



class ComparisonPerformancesByTemplateSimilarity(BaseWidget):
    """
    Plot run times for a study.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    similarity_matrix: matrix
        The similarity between the templates in the gt recording and the ones
        found by the sorter
    ax: matplotlib ax
        The ax to be used. If not given a figure is created

    """
    def __init__(self, gt_comp, similarity_matrix, ax=None, ylim=(0.6, 1)):

        self.gt_comp = gt_comp
        self.similarity_matrix = similarity_matrix
        self.ylim = ylim

        BaseWidget.__init__(self, ax=ax)

    def plot(self):


        all_results = {'similarity' : [], 'accuracy' : []}
        comp = self.gt_comp

        for i, u1 in enumerate(comp.sorting1.unit_ids):
            u2 = comp.best_match_12[u1]
            if u2 != -1:
                all_results['similarity'] += [self.similarity_matrix[comp.sorting1.id_to_index(u1), comp.sorting2.id_to_index(u2)]]
                all_results['accuracy'] += [comp.agreement_scores.at[u1, u2]]

        all_results['similarity'] = np.array(all_results['similarity'])
        all_results['accuracy'] = np.array(all_results['accuracy'])

        self.ax.plot(all_results['similarity'], all_results['accuracy'], '.')

        self.ax.set_ylabel('accuracy')
        self.ax.set_xlabel('cosine similarity')
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)


def plot_gt_performances(*args, **kwargs):
    W = ComparisonPerformancesWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_performances.__doc__ = ComparisonPerformancesWidget.__doc__

def plot_gt_performances_averages(*args, **kwargs):
    W = ComparisonPerformancesAveragesWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_performances_averages.__doc__ = ComparisonPerformancesAveragesWidget.__doc__


def plot_gt_performances_by_template_similarity(*args, **kwargs):
    W = ComparisonPerformancesByTemplateSimilarity(*args, **kwargs)
    W.plot()
    return W
plot_gt_performances_by_template_similarity.__doc__ = ComparisonPerformancesByTemplateSimilarity.__doc__
