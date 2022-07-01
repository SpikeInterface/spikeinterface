import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors

from .basewidget import BaseWidget
from spikeinterface.comparison.collisioncomparison import CollisionGTComparison

class ComparisonCollisionPairByPairWidget(BaseWidget):
    """
    Plots CollisionGTComparison pair by pair.

    Parameters
    ----------
    comp: CollisionGTComparison
        The collision ground truth comparison object
    unit_ids: list
        List of considered units
    nbins: int
        Number of bins
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created

    Returns
    -------
    W: MultiCompGraphWidget
        The output widget
    """
    def __init__(self, comp, unit_ids=None, figure=None, ax=None):

        BaseWidget.__init__(self, figure, ax)
        if unit_ids is None:
            # take all units
            unit_ids = comp.sorting1.get_unit_ids()

        self.comp = comp
        self.unit_ids = unit_ids

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        fig = self.figure

        for ax in fig.axes:
            ax.remove()

        n = len(self.unit_ids)
        gs = gridspec.GridSpec(ncols=n, nrows=n, figure=fig)

        axs = np.empty((n, n), dtype=object)
        ax = None
        for r in range(n):
            for c in range(n):
                ax = fig.add_subplot(gs[r, c], sharex=ax, sharey=ax)
                if c > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if r < n - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                axs[r, c] = ax

        fs = self.comp.sorting1.get_sampling_frequency()

        lags = self.comp.bins / fs * 1000
        width = lags[1] - lags[0]

        for r in range(n):
            for c in range(r+1, n):

                ax = axs[r, c]

                u1 = self.unit_ids[r]
                u2 = self.unit_ids[c]
                ind1 = self.comp.sorting1.id_to_index(u1)
                ind2 = self.comp.sorting1.id_to_index(u2)

                tp = self.comp.all_tp[ind1, ind2, :]
                fn = self.comp.all_fn[ind1, ind2, :]
                ax.bar(lags[:-1], tp, width=width,  color='g', align='edge')
                ax.bar(lags[:-1], fn, width=width, bottom=tp, color='r', align='edge')

                ax = axs[c, r]
                tp = self.comp.all_tp[ind2, ind1, :]
                fn = self.comp.all_fn[ind2, ind1, :]
                ax.bar(lags[:-1], tp, width=width,  color='g', align='edge')
                ax.bar(lags[:-1], fn, width=width, bottom=tp, color='r', align='edge')

        for r in range(n):
            ax = axs[r, 0]
            u1 = self.unit_ids[r]
            ax.set_ylabel(f'gt id{u1}')

        for c in range(n):
            ax = axs[0, c]
            u2 = self.unit_ids[c]
            ax.set_title(f'collision with \ngt id{u2}')

        ax = axs[-1, 0]
        ax.set_xlabel('collision lag [ms]')


class ComparisonCollisionBySimilarityWidget(BaseWidget):
    """
    Plots CollisionGTComparison pair by pair orderer by cosine_similarity

    Parameters
    ----------
    comp: CollisionGTComparison
        The collision ground truth comparison object
    templates: array
        template of units
    mode: 'heatmap' or 'lines'
        to see collision curves for every pairs ('heatmap') or as lines averaged over pairs. 
    similarity_bins: array
        if mode is 'lines', the bins used to average the pairs
    cmap: string
        colormap used to show averages if mode is 'lines'
    metric: 'cosine_similarity'
        metric for ordering
    good_only: True
        keep only the pairs with a non zero accuracy (found templates)
    min_accuracy: float
        If good only, the minimum accuracy every cell should have, individually, to be
        considered in a putative pair
    unit_ids: list
        List of considered units
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created
    """

    def __init__(self, comp, templates, unit_ids=None, metric='cosine_similarity', figure=None, ax=None, 
        mode='heatmap', similarity_bins=np.linspace(-0.4, 1, 8), cmap='winter', good_only=True,  min_accuracy=0.9, show_legend=False,
        ylim = (0, 1)):
        BaseWidget.__init__(self, figure, ax)

        assert mode in ['heatmap', 'lines']

        if unit_ids is None:
            # take all units
            unit_ids = comp.sorting1.get_unit_ids()

        self.comp = comp
        self.cmap = cmap
        self.mode = mode
        self.ylim = ylim
        self.show_legend = show_legend
        self.similarity_bins = similarity_bins
        self.templates = templates
        self.unit_ids = unit_ids
        self.metric = metric
        self.good_only = good_only
        self.min_accuracy = min_accuracy

    def plot(self):
        self._do_plot()

    def _do_plot(self):

        import sklearn

        # compute similarity
        # take index of template (respect unit_ids order)
        all_unit_ids = list(self.comp.sorting1.get_unit_ids())
        template_inds = [all_unit_ids.index(u) for u in self.unit_ids]

        templates = self.templates[template_inds, :, :].copy()
        flat_templates = templates.reshape(templates.shape[0], -1)
        if self.metric == 'cosine_similarity':
            similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
        else:
            raise NotImplementedError('metric=...')

        fs = self.comp.sorting1.get_sampling_frequency()
        lags = self.comp.bins / fs * 1000

        n = len(self.unit_ids)

        similarities, recall_scores, pair_names = self.comp.compute_collision_by_similarity(similarity_matrix, unit_ids=self.unit_ids, good_only=self.good_only, min_accuracy=self.min_accuracy)

        if self.mode == 'heatmap':

            fig = self.figure
            for ax in fig.axes:
                ax.remove()

            n_pair = len(similarities)

            ax0 = fig.add_axes([0.1, 0.1, .25, 0.8])
            ax1 = fig.add_axes([0.4, 0.1, .5, 0.8], sharey=ax0)

            plt.setp(ax1.get_yticklabels(), visible=False)

            im = ax1.imshow(recall_scores[::-1, :],
                        cmap='viridis',
                        aspect='auto',
                        interpolation='none',
                        extent=(lags[0], lags[-1], -0.5, n_pair-0.5),
                        )
            im.set_clim(0,1)

            ax0.plot(similarities, np.arange(n_pair), color='k')

            ax0.set_yticks(np.arange(n_pair))
            ax0.set_yticklabels(pair_names)
            # ax0.set_xlim(0,1)

            ax0.set_xlabel(self.metric)
            ax0.set_ylabel('pairs')

            ax1.set_xlabel('lag (ms)')
        elif self.mode == 'lines':
            my_cmap = plt.get_cmap(self.cmap)
            cNorm  = matplotlib.colors.Normalize(vmin=self.similarity_bins.min(), vmax=self.similarity_bins.max())
            scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

            # plot by similarity bins
            if self.ax is None:
                fig, ax = plt.subplots()
            else:
                ax = self.ax
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            order = np.argsort(similarities)
            similarities = similarities[order]
            recall_scores = recall_scores[order, :]

            for i in range(self.similarity_bins.size - 1):
                cmin, cmax = self.similarity_bins[i], self.similarity_bins[i + 1]

                amin, amax = np.searchsorted(similarities, [cmin, cmax])
                mean_recall_scores = np.nanmean(recall_scores[amin:amax], axis=0)

                colorVal = scalarMap.to_rgba((cmin+cmax)/2)
                ax.plot(lags[:-1] + (lags[1]-lags[0]) / 2, mean_recall_scores, label='CS in [%g,%g]' %(cmin, cmax), c=colorVal)

            if self.show_legend:
                ax.legend()
            ax.set_ylim(self.ylim)
            ax.set_xlabel('lags (ms)')
            ax.set_ylabel('collision accuracy')      



class StudyComparisonCollisionBySimilarityWidget(BaseWidget):


    def __init__(self, study, metric='cosine_similarity',
                 similarity_bins=np.linspace(-0.4, 1, 8), show_legend=False, ylim=(0.5, 1),
                 good_only=True,
                 min_accuracy=0.9,
                 ncols=3, axes=None, cmap='winter'):

        if axes is None:
            num_axes = len(study.sorter_names)
        else:
            num_axes = None
        BaseWidget.__init__(self, None, None, axes, ncols=ncols, num_axes=num_axes)

        self.ncols = ncols
        self.study = study
        self.metric = metric
        self.cmap = cmap
        self.similarity_bins = np.asarray(similarity_bins)
        self.show_legend = show_legend
        self.ylim = ylim
        self.good_only = good_only
        self.min_accuracy = min_accuracy


    def plot(self):

        my_cmap = plt.get_cmap(self.cmap)
        cNorm  = matplotlib.colors.Normalize(vmin=self.similarity_bins.min(), vmax=self.similarity_bins.max())
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)
        self.study.precompute_scores_by_similarities(self.good_only, min_accuracy=self.min_accuracy)
        lags = self.study.get_lags()

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):

            curves = self.study.get_lag_profile_over_similarity_bins(self.similarity_bins, sorter_name)

            # plot by similarity bins
            ax = self.axes.flatten()[sorter_ind]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for i in range(self.similarity_bins.size - 1):
                cmin, cmax = self.similarity_bins[i], self.similarity_bins[i + 1]
                colorVal = scalarMap.to_rgba((cmin+cmax)/2)
                ax.plot(lags[:-1] + (lags[1]-lags[0]) / 2, curves[(cmin, cmax)], label='CS in [%g,%g]' %(cmin, cmax), c=colorVal)

            if np.mod(sorter_ind, self.ncols) == 0:
                ax.set_ylabel('collision accuracy')

            if sorter_ind > (len(self.study.sorter_names) // self.ncols):
                ax.set_xlabel('lags (ms)')

            ax.set_title(sorter_name)
            if self.show_legend:
                ax.legend()

            if self.ylim is not None:
                ax.set_ylim(self.ylim)



class StudyComparisonCollisionBySimilarityRangeWidget(BaseWidget):


    def __init__(self, study, metric='cosine_similarity',
                 similarity_range=[0, 1], show_legend=False, ylim=(0.5, 1),
                 good_only=True, min_accuracy=0.9, ax=None):

        BaseWidget.__init__(self, None, ax)

        self.study = study
        self.metric = metric
        self.similarity_range = similarity_range
        self.show_legend = show_legend
        self.ylim = ylim
        self.good_only = good_only
        self.min_accuracy = min_accuracy


    def plot(self):

        self.study.precompute_scores_by_similarities(self.good_only, min_accuracy=self.min_accuracy)
        lags = self.study.get_lags()

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):

            mean_recall_scores = self.study.get_mean_over_similarity_range(self.similarity_range, sorter_name)
            self.ax.plot(lags[:-1] + (lags[1]-lags[0]) / 2, mean_recall_scores, label=sorter_name, c='C%d' %sorter_ind)

        self.ax.set_ylabel('collision accuracy')
        self.ax.set_xlabel('lags (ms)')

        if self.show_legend:
            self.ax.legend()

        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)


class StudyComparisonCollisionBySimilarityRangesWidget(BaseWidget):


    def __init__(self, study, metric='cosine_similarity',
                 similarity_ranges=np.linspace(-0.4, 1, 8), show_legend=False, ylim=(0.5, 1),
                 good_only=True, min_accuracy=0.9, ax=None, show_std=False):

        BaseWidget.__init__(self, None, ax)

        self.study = study
        self.metric = metric
        self.similarity_ranges = similarity_ranges
        self.show_legend = show_legend
        self.ylim = ylim
        self.good_only = good_only
        self.show_std = show_std
        self.min_accuracy = min_accuracy


    def plot(self):

        self.study.precompute_scores_by_similarities(self.good_only, min_accuracy=self.min_accuracy)
        lags = self.study.get_lags()

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):

            all_similarities = self.study.all_similarities[sorter_name]
            all_recall_scores = self.study.all_recall_scores[sorter_name]

            order = np.argsort(all_similarities)
            all_similarities = all_similarities[order]
            all_recall_scores = all_recall_scores[order, :]

            mean_recall_scores = []
            std_recall_scores = []
            for i in range(self.similarity_ranges.size - 1):
                cmin, cmax = self.similarity_ranges[i], self.similarity_ranges[i + 1]
                amin, amax = np.searchsorted(all_similarities, [cmin, cmax])
                mean_recall_scores += [np.nanmean(all_recall_scores[amin:amax])]
                std_recall_scores += [np.nanstd(all_recall_scores[amin:amax])]

            xaxis = np.diff(self.similarity_ranges)/2 + self.similarity_ranges[:-1]

            if not self.show_std:
                self.ax.plot(xaxis, mean_recall_scores, label=sorter_name, c='C%d' %sorter_ind)
            else:
                self.ax.errorbar(xaxis, mean_recall_scores, yerr=std_recall_scores, label=sorter_name, c='C%d' %sorter_ind)

        self.ax.set_ylabel('collision accuracy')
        self.ax.set_xlabel('similarity')

        if self.show_legend:
            self.ax.legend()

        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)


def plot_comparison_collision_pair_by_pair(*args, **kwargs):
    W = ComparisonCollisionPairByPairWidget(*args, **kwargs)
    W.plot()
    return W
plot_comparison_collision_pair_by_pair.__doc__ = ComparisonCollisionPairByPairWidget.__doc__


def plot_comparison_collision_by_similarity(*args, **kwargs):
    W = ComparisonCollisionBySimilarityWidget(*args, **kwargs)
    W.plot()
    return W
plot_comparison_collision_by_similarity.__doc__ = ComparisonCollisionBySimilarityWidget.__doc__


def plot_study_comparison_collision_by_similarity(*args, **kwargs):
    W = StudyComparisonCollisionBySimilarityWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_collision_by_similarity.__doc__ = StudyComparisonCollisionBySimilarityWidget.__doc__

def plot_study_comparison_collision_by_similarity_range(*args, **kwargs):
    W = StudyComparisonCollisionBySimilarityRangeWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_collision_by_similarity_range.__doc__ = StudyComparisonCollisionBySimilarityRangeWidget.__doc__

def plot_study_comparison_collision_by_similarity_ranges(*args, **kwargs):
    W = StudyComparisonCollisionBySimilarityRangesWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_collision_by_similarity_ranges.__doc__ = StudyComparisonCollisionBySimilarityRangesWidget.__doc__
