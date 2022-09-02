import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors


class StudyComparisonCorrelogramBySimilarityWidget(BaseWidget):

    def __init__(self, study, metric='cosine_similarity',
                 similarity_bins=np.linspace(-0.4, 1, 8), show_legend=False,
                 ncols=3, axes=None, cmap='winter', ylim=(0,0.5)):

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
        self.ylim = ylim

    def plot(self):

        my_cmap = plt.get_cmap(self.cmap)
        cNorm  = matplotlib.colors.Normalize(vmin=self.similarity_bins.min(), vmax=self.similarity_bins.max())
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

        self.study.precompute_scores_by_similarities()
        time_bins = self.study.time_bins

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):

            result = self.study.get_error_profile_over_similarity_bins(self.similarity_bins, sorter_name)

            # plot by similarity bins
            ax = self.axes.flatten()[sorter_ind]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for i in range(self.similarity_bins.size - 1):
                cmin, cmax = self.similarity_bins[i], self.similarity_bins[i + 1]
                colorVal = scalarMap.to_rgba((cmin+cmax)/2)
                ax.plot(time_bins, result[(cmin, cmax)], label='CS in [%g,%g]' %(cmin, cmax), c=colorVal)

            if np.mod(sorter_ind, self.ncols) == 0:
                ax.set_ylabel('cc error')

            if sorter_ind >= (len(self.study.sorter_names) // self.ncols):
                ax.set_xlabel('lags (ms)')

            ax.set_title(sorter_name)
            if self.show_legend:
                ax.legend()

            if self.ylim is not None:
                ax.set_ylim(self.ylim)


class StudyComparisonCorrelogramBySimilarityRangesMeanErrorWidget(BaseWidget):


    def __init__(self, study, metric='cosine_similarity',
                 similarity_ranges=np.linspace(-0.4, 1, 8), show_legend=False,
                 ax=None, show_std=False, ylim=(0,0.5)):

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
                mean_rerrors += [np.nanmean(all_errors[amin:amax])]
                std_errors += [np.nanstd(all_errors[amin:amax])]

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


def plot_study_comparison_correlogram_by_similarity(*args, **kwargs):
    W = StudyComparisonCorrelogramBySimilarityWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_correlogram_by_similarity.__doc__ = StudyComparisonCorrelogramBySimilarityWidget.__doc__

# def plot_study_comparison_Correlogram_by_similarity_range(*args, **kwargs):
#     W = StudyComparisonCorrelogramBySimilarityRangeWidget(*args, **kwargs)
#     W.plot()
#     return W
# plot_study_comparison_Correlogram_by_similarity_range.__doc__ = StudyComparisonCorrelogramBySimilarityRangeWidget.__doc__


def plot_study_comparison_correlogram_by_similarity_ranges_mean_error(*args, **kwargs):
    W = StudyComparisonCorrelogramBySimilarityRangesMeanErrorWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_correlogram_by_similarity_ranges_mean_error.__doc__ = StudyComparisonCorrelogramBySimilarityRangesMeanErrorWidget.__doc__
