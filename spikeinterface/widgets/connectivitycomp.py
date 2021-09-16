import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors


class StudyComparisonConnectivityBySimilarityWidget(BaseWidget):


    def __init__(self, study, metric='cosine_similarity', 
                        similarity_bins=np.linspace(0, 1, 11), show_legend=False,
                        ncols=3, axes=None, cmap='winter'):
        
        if axes is None:
            num_axes = len(study.sorter_names)
        else:
            num_axes = None
        BaseWidget.__init__(self, None, None, axes, ncols=ncols, num_axes=num_axes)

        self.ncols = ncols
        self.cmap = cmap
        self.study = study
        self.metric = metric
        self.similarity_bins = np.asarray(similarity_bins)
        self.show_legend = show_legend

    def plot(self):

        my_cmap = plt.get_cmap(self.cmap)
        cNorm  = matplotlib.colors.Normalize(vmin=self.similarity_bins.min(), vmax=self.similarity_bins.max())
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

        self.study.precompute_scores_by_similarities()
        time_axis = self.study.time_axis

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):
            
            result = self.get_error_profile_over_similarity_bins(similarity_bins, sorter_name)

            # plot by similarity bins
            ax = self.axes.flatten()[sorter_ind]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for i in range(self.similarity_bins.size - 1):
                cmin, cmax = self.similarity_bins[i], self.similarity_bins[i + 1]
                amin, amax = np.searchsorted(all_similarities, [cmin, cmax])
                colorVal = scalarMap.to_rgba((cmin+cmax)/2)
                ax.plot(time_axis, result[(cmin, cmax)], label='$CC \in [%g,%g]$' %(cmin, cmax), c=colorVal)
            
            if np.mod(sorter_ind, self.ncols) == 0:
                ax.set_ylabel('cc error')

            if sorter_ind >= (len(self.study.sorter_names) // self.ncols):
                ax.set_xlabel('lags (ms)')

            ax.set_title(sorter_name)
            if self.show_legend:
                ax.legend()

            #if self.ylim is not None:
            #    ax.set_ylim(self.ylim)


class StudyComparisonConnectivityBySimilarityRangesMeanErrorWidget(BaseWidget):


    def __init__(self, study, metric='cosine_similarity', 
                        similarity_ranges=np.linspace(0, 1, 11), show_legend=False,
                        ax=None, show_std=False, ylim = None):
        
        BaseWidget.__init__(self, None, ax)

        self.study = study
        self.metric = metric
        self.show_std = show_std
        self.ylim = None
        self.similarity_ranges = np.asarray(similarity_ranges)
        self.show_legend = show_legend

    def plot(self):

        self.study.precompute_scores_by_similarities()

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):
            
            
            all_similarities = self.study.all_similarities[sorter_name]
            all_errors = self.study.all_errors[sorter_name]
            
            order = np.argsort(all_similarities)
            all_similarities = all_similarities[order]
            all_errors = all_errors[order, :]

            mean_rerrors = []
            std_errors = []
            for i in range(self.similarity_ranges.size - 1):
                cmin, cmax = self.similarity_ranges[i], self.similarity_ranges[i + 1]
                amin, amax = np.searchsorted(all_similarities, [cmin, cmax])
                value = np.mean(all_errors[amin:amax])
                mean_rerrors += [np.nan_to_num(value)]
                value = np.std(all_errors[amin:amax])
                std_errors += [np.nan_to_num(value)]
            
            xaxis = np.diff(self.similarity_ranges)/2 + self.similarity_ranges[:-1]

            if not self.show_std:
                self.ax.plot(xaxis, mean_rerrors, label=sorter_name, c='C%d' %sorter_ind)
            else:
                self.ax.errorbar(xaxis, mean_rerrors, yerr=std_errors, label=sorter_name, c='C%d' %sorter_ind)

        self.ax.set_ylabel('cc error')
        self.ax.set_xlabel('similarity')

        if self.show_legend:
            self.ax.legend()

        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)


def plot_study_comparison_connectivity_by_similarity(*args, **kwargs):
    W = StudyComparisonConnectivityBySimilarityWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_connectivity_by_similarity.__doc__ = StudyComparisonConnectivityBySimilarityWidget.__doc__

# def plot_study_comparison_connectivity_by_similarity_range(*args, **kwargs):
#     W = StudyComparisonConnectivityBySimilarityRangeWidget(*args, **kwargs)
#     W.plot()
#     return W
# plot_study_comparison_connectivity_by_similarity_range.__doc__ = StudyComparisonConnectivityBySimilarityRangeWidget.__doc__


def plot_study_comparison_connectivity_by_similarity_ranges_mean_error(*args, **kwargs):
    W = StudyComparisonConnectivityBySimilarityRangesMeanErrorWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_connectivity_by_similarity_ranges_mean_error.__doc__ = StudyComparisonConnectivityBySimilarityRangesMeanErrorWidget.__doc__