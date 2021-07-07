import numpy as np


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from .basewidget import BaseWidget, BaseMultiWidget
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
    def __init__(self, comp, unit_ids=None, nbins=11, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)
        if unit_ids is None:
            # take all units
            unit_ids = comp.sorting1.get_unit_ids()

        self.comp = comp
        self.unit_ids = unit_ids
        self.nbins = nbins
        fs = self.comp.sorting1.get_sampling_frequency()
        self.width = (2*comp.collision_lag) / fs * 1000.

        self._compute()

    def _compute(self):
        
        fs = self.comp.sorting1.get_sampling_frequency()
        all_tp_count1 = []
        all_fn_count1 = []
        all_tp_count2 = []
        all_fn_count2 = []

        n = len(self.unit_ids)

        for r in range(n):
            for c in range(r+1, n):
                
                u1 = self.unit_ids[r]
                u2 = self.unit_ids[c]
                
                bins, tp_count1, fn_count1, tp_count2, fn_count2 = self.comp.get_label_count_per_collision_bins(u1, u2, nbins=self.nbins)
                self.lags = bins[:-1] / fs * 1000
                all_tp_count1 += [tp_count1]
                all_tp_count2 += [tp_count2]
                all_fn_count1 += [fn_count1]
                all_fn_count2 += [fn_count2]


        self.all_fn_count1 = np.array(all_fn_count1)
        self.all_fn_count2 = np.array(all_fn_count2)
        self.all_tp_count1 = np.array(all_tp_count1)
        self.all_tp_count2 = np.array(all_tp_count2)


    def plot(self):
        self._do_plot()

    def _do_plot(self):
        fig = self.figure
        
        for ax in fig.axes:
            ax.remove()
        
        n = len(self.unit_ids)
        gs = gridspec.GridSpec(ncols=n, nrows=n, figure=fig)
        
        axs = np.empty((n,n), dtype=object)
        ax = None
        for r in range(n):
            for c in range(n):
                ax = fig.add_subplot(gs[r, c], sharex=ax, sharey=ax)
                if c > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if r < n-1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                axs[r, c] = ax
        
        fs = self.comp.sorting1.get_sampling_frequency()
        count = 0

        for r in range(n):
            for c in range(r+1, n):
                                
                ax = axs[r, c]
                ax.bar(self.lags, self.all_tp_count1[count], width=self.width,  color='g')
                ax.bar(self.lags, self.all_fn_count1[count], width=self.width, bottom=self.all_tp_count1[count], color='r')
                
                ax = axs[c, r]
                ax.bar(self.lags, self.all_tp_count2[count], width=self.width,  color='g')
                ax.bar(self.lags, self.all_fn_count2[count], width=self.width, bottom=self.all_tp_count2[count], color='r')
                count =+ 1
        
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
        self._compute()

    def _compute(self):
        import sklearn
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
            for c in range(r+1, n):
                
                u1 = self.unit_ids[r]
                u2 = self.unit_ids[c]
                
                bins, tp_count1, fn_count1, tp_count2, fn_count2 = self.comp.get_label_count_per_collision_bins(u1, u2, nbins=self.nbins)
                self.lags = bins[:-1] / fs * 1000
                
                accuracy1 = tp_count1 / (tp_count1 + fn_count1)
                recall_scores.append(accuracy1)
                similarities.append(similarity_matrix[r, c])
                pair_names.append(f'{u1} {u2}')
                
                accuracy2 = tp_count2 / (tp_count2 + fn_count2)
                recall_scores.append(accuracy2)
                similarities.append(similarity_matrix[r, c])
                pair_names.append(f'{u2} {u1}')

        self.recall_scores = np.array(recall_scores)
        self.similarities = np.array(similarities)
        self.pair_names = np.array(pair_names)
        
        order = np.argsort(self.similarities)
        self.similarities = self.similarities[order]
        self.recall_scores = self.recall_scores[order, :]
        self.pair_names = self.pair_names[order]


    def get_good_only(self):
        valid_indices = np.where(self.recall_scores.sum(1) > 0)[0]
        return self.similarities[valid_indices], self.recall_scores[valid_indices], self.pair_names[valid_indices]

    def plot(self, good_only=False):
        self._do_plot(good_only)

    def _do_plot(self, good_only):
        
        fig = self.figure

        if good_only:
            similarities, scores, names = self.get_good_only()
        else:
            similarities, scores, names = self.similarities, self.recall_scores, self.pair_names

        for ax in fig.axes:
            ax.remove()
        
        # plot
        n_pair = len(similarities)
        
        ax0 = fig.add_axes([0.1 , 0.1 , .25 , 0.8 ] )
        ax1 = fig.add_axes([0.4 , 0.1 , .5 , 0.8 ] , sharey=ax0)
        
        plt.setp(ax1.get_yticklabels(), visible=False)
        
        im = ax1.imshow(scores[::-1, :],
                    cmap='viridis',
                    aspect='auto',
                    interpolation='none',
                    extent=(self.lags[0], self.lags[-1], -0.5, n_pair-0.5),
                    )
        im.set_clim(0,1)
        
        ax0.plot(similarities, np.arange(n_pair), color='k')
        
        ax0.set_yticks(np.arange(n_pair))
        ax0.set_yticklabels(names)
        # ax0.set_xlim(0,1)
        
        ax0.set_xlabel(self.metric)
        ax0.set_ylabel('pairs')
        
        ax1.set_xlabel('lag [ms]')



class StudyComparisonCollisionBySimilarityWidget(BaseMultiWidget):


    def __init__(self, study, metric='cosine_similarity', collision_lag=2, exhaustive_gt=False, nbins=10, figure=None, ax=None, axes=None):
        
        self._ncols = 3
        self._nrows = len(study.sorter_names) % self._ncols
        
        if axes is None and ax is None:
            figure, axes = plt.subplots(nrows=self._nrows, ncols=self._ncols, sharex=True, sharey=True)

        BaseMultiWidget.__init__(self, figure, ax, axes)

        self.study = study
        self.collision_lag = collision_lag
        self.exhaustive_gt = exhaustive_gt
        self.nbins = nbins
        self.metric = metric
        self.all_results = {}
        self._compute()
        

    def _compute(self):

        for sort_name in self.study.sorter_names:
            self.all_results[sort_name] = {}
            for rec_name in self.study.rec_names:        
                gt_sorting = self.study.get_ground_truth(rec_name)
                tested_sorting = self.study.get_sorting(sort_name, rec_name)

                comp = CollisionGTComparison(gt_sorting, tested_sorting, collision_lag=self.collision_lag, exhaustive_gt=self.exhaustive_gt)
                templates = self.study.get_templates(rec_name)
                widget = ComparisonCollisionBySimilarityWidget(comp, templates)
                self.lags = widget.lags

                similarities, data, pair_names = widget.get_good_only()

                if 'similarity' in self.all_results[sort_name]:
                    self.all_results[sort_name]['similarity'] = np.concatenate((self.all_results[sort_name]['similarity'], similarities))
                    self.all_results[sort_name]['data'] = np.vstack((self.all_results[sort_name]['data'], data))
                    self.all_results[sort_name]['pair'] = np.concatenate((self.all_results[sort_name]['pair'], pair_names))
                else:   
                    self.all_results[sort_name]['similarity'] = similarities
                    self.all_results[sort_name]['data'] = data
                    self.all_results[sort_name]['pair'] = pair_names
                

    def plot(self, cc_similarity=np.arange(0, 1, 0.1), show_legend=False, ylim=None):

        import matplotlib.colors as colors

        my_cmap = plt.get_cmap('winter')
        cNorm  = colors.Normalize(vmin=cc_similarity.min(), vmax=cc_similarity.max())
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

        for scount, sort_name in enumerate(self.study.sorter_names):

            ax = self.get_tiled_ax(scount, self._nrows, self._ncols)

            for count, i in enumerate(range(len(cc_similarity) - 1)):
                cmin, cmax = cc_similarity[i], cc_similarity[i + 1]
                amin, amax = np.searchsorted(self.all_results[sort_name]['similarity'], [cmin, cmax])
                r = self.all_results[sort_name]['data'][amin:amax]
                colorVal = scalarMap.to_rgba((cmin+cmax)/2)
                ax.plot(self.lags, np.nan_to_num(r.mean(0)), label='$CC \in [%g,%g]$' %(cmin, cmax), c=colorVal)
                ax.set_title(sort_name)
                if show_legend:
                    ax.legend()
                if np.mod(scount, 3) == 0:
                    ax.set_ylabel('collision accuracy')
                else:
                    ax.tick_params(labelleft=False)

                if self._nrows > 1 and (scount < self._nrows*self._ncols):
                    ax.set_xlabel('lag (ms)')
                else:
                    ax.tick_params(labelbottom=False)

                if ylim is not None:
                    ax.set_ylim(ylim)

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



