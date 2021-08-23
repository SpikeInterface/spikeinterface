import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from .basewidget import BaseWidget


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

    def __init__(self, comp, unit_ids=None, nbins=10, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        if unit_ids is None:
            # take all units
            unit_ids = comp.sorting1.get_unit_ids()

        self.comp = comp
        self.unit_ids = unit_ids
        self.nbins = nbins

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

        for r in range(n):
            for c in range(r + 1, n):
                u1 = self.unit_ids[r]
                u2 = self.unit_ids[c]

                bins, tp_count1, fn_count1, tp_count2, fn_count2 = self.comp.get_label_count_per_collision_bins(u1, u2,
                                                                                                                nbins=self.nbins)

                width = (bins[1] - bins[0]) / fs * 1000.
                lags = bins[:-1] / fs * 1000

                ax = axs[r, c]
                ax.bar(lags, tp_count1, width=width, color='g')
                ax.bar(lags, fn_count1, width=width, bottom=tp_count1, color='r')

                ax = axs[c, r]
                ax.bar(lags, tp_count2, width=width, color='g')
                ax.bar(lags, fn_count2, width=width, bottom=tp_count2, color='r')

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
    metric: cosine_similarity',
        metric for ordering
    unit_ids: list
        List of considered units
    nbins: int
        Number of bins
    figure: matplotlib figure
        The figure to be used. If not given a figure is created
    ax: matplotlib axis
        The axis to be used. If not given an axis is created
    """

    def __init__(self, comp, templates, unit_ids=None, metric='cosine_similarity', nbins=10, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        if unit_ids is None:
            # take all units
            unit_ids = comp.sorting1.get_unit_ids()

        self.comp = comp
        self.templates = templates
        self.unit_ids = unit_ids
        self.nbins = nbins
        self.metric = metric

    def plot(self):
        self._do_plot()

    def _do_plot(self):
        import sklearn

        fig = self.figure

        for ax in fig.axes:
            ax.remove()

        # compute similarity
        # take index of temmplate (respect unit_ids order)
        all_unit_ids = list(self.comp.sorting1.get_unit_ids())
        template_inds = [all_unit_ids.index(u) for u in self.unit_ids]

        templates = self.templates[template_inds, :, :].copy()
        flat_templates = templates.reshape(templates.shape[0], -1)
        if self.metric == 'cosine_similarity':
            similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
        else:
            raise NotImplementedError('metric=...')

        # print(similarity_matrix)

        n = len(self.unit_ids)

        fs = self.comp.sorting1.get_sampling_frequency()
        recall_scores = []
        similarities = []
        pair_names = []
        for r in range(n):
            for c in range(r + 1, n):
                u1 = self.unit_ids[r]
                u2 = self.unit_ids[c]

                bins, tp_count1, fn_count1, tp_count2, fn_count2 = self.comp.get_label_count_per_collision_bins(u1, u2,
                                                                                                                nbins=self.nbins)

                width = (bins[1] - bins[0]) / fs * 1000.
                lags = bins[:-1] / fs * 1000

                accuracy1 = tp_count1 / (tp_count1 + fn_count1)
                recall_scores.append(accuracy1)
                similarities.append(similarity_matrix[r, c])
                pair_names.append(f'{u1} {u2}')

                accuracy2 = tp_count2 / (tp_count2 + fn_count2)
                recall_scores.append(accuracy2)
                similarities.append(similarity_matrix[r, c])
                pair_names.append(f'{u2} {u1}')

        recall_scores = np.array(recall_scores)
        similarities = np.array(similarities)
        pair_names = np.array(pair_names)

        order = np.argsort(similarities)
        similarities = similarities[order]
        recall_scores = recall_scores[order, :]
        pair_names = pair_names[order]

        #  plot
        n_pair = len(similarities)

        ax0 = fig.add_axes([0.1, 0.1, .25, 0.8])
        ax1 = fig.add_axes([0.4, 0.1, .5, 0.8], sharey=ax0)

        plt.setp(ax1.get_yticklabels(), visible=False)

        im = ax1.imshow(recall_scores[::-1, :],
                        cmap='viridis',
                        aspect='auto',
                        interpolation='none',
                        extent=(lags[0], lags[-1], -0.5, n_pair - 0.5),
                        )
        im.set_clim(0, 1)

        ax0.plot(similarities, np.arange(n_pair), color='k')

        ax0.set_yticks(np.arange(n_pair))
        ax0.set_yticklabels(pair_names)
        # ax0.set_xlim(0,1)

        ax0.set_xlabel(self.metric)
        ax0.set_ylabel('pairs')

        ax1.set_xlabel('lag [ms]')


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
