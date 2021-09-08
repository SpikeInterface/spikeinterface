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

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):
            
            # loop over recordings
            all_errors = []
            all_similarities = []
            for rec_name in self.study.rec_names:
                
                templates = self.study.get_templates(rec_name)
                flat_templates = templates.reshape(templates.shape[0], -1)
                import sklearn
                similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
            
                comp = self.study.comparisons[(rec_name, sorter_name)]
                similarities, errors = comp.compute_connectivity_by_similarity(similarity_matrix)

                all_similarities.append(similarities)
                all_errors.append(errors)
            
            all_similarities = np.concatenate(all_similarities, axis=0)
            all_errors = np.concatenate(all_errors, axis=0)
            
            order = np.argsort(all_similarities)
            all_similarities = all_similarities[order]
            all_errors = all_errors[order, :]

            
            # plot by similarity bins
            ax = self.axes.flatten()[sorter_ind]
            
            for i in range(self.similarity_bins.size - 1):
                cmin, cmax = self.similarity_bins[i], self.similarity_bins[i + 1]
                amin, amax = np.searchsorted(all_similarities, [cmin, cmax])
                mean_error = np.mean(all_errors[amin:amax], axis=0)
                mean_error = np.nan_to_num(mean_error)
                colorVal = scalarMap.to_rgba((cmin+cmax)/2)
                ax.plot(comp.time_axis, mean_error, label='$CC \in [%g,%g]$' %(cmin, cmax), c=colorVal)
            
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

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):
            
            # loop over recordings
            all_errors = []
            all_similarities = []
            for rec_name in self.study.rec_names:
                
                templates = self.study.get_templates(rec_name)
                flat_templates = templates.reshape(templates.shape[0], -1)
                import sklearn
                similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
            
                comp = self.study.comparisons[(rec_name, sorter_name)]
                similarities, errors = comp.compute_connectivity_by_similarity(similarity_matrix)

                all_similarities.append(similarities)
                all_errors.append(errors)
            
            all_similarities = np.concatenate(all_similarities, axis=0)
            all_errors = np.concatenate(all_errors, axis=0)
            
            order = np.argsort(all_similarities)
            all_similarities = all_similarities[order]
            all_errors = all_errors[order, :]

            
            # plot by similarity bins
            ax = self.axes.flatten()[sorter_ind]
            
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
                self.ax.errobar(xaxis, mean_rerrors, yerr=std_errors, label=sorter_name, c='C%d' %sorter_ind)

        self.ax.set_ylabel('cc error')
        self.ax.set_xlabel('similarity')

        if self.show_legend:
            self.ax.legend()

        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)

            #if self.ylim is not None:
            #    ax.set_ylim(self.ylim)


class StudyComparisonConnectivityBySimilarityRangeWidget(BaseWidget):


    def __init__(self, study, metric='cosine_similarity', 
                        similarity_range=[0, 1], show_legend=False,
                        ax=None):
        
        BaseWidget.__init__(self, None, ax)

        self.study = study
        self.metric = metric
        self.similarity_range = similarity_range
        self.show_legend = show_legend

    def plot(self):

        for sorter_ind, sorter_name in enumerate(self.study.sorter_names):
            
            # loop over recordings
            all_errors = []
            all_similarities = []
            for rec_name in self.study.rec_names:
                
                templates = self.study.get_templates(rec_name)
                flat_templates = templates.reshape(templates.shape[0], -1)
                import sklearn
                similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(flat_templates)
            
                comp = self.study.comparisons[(rec_name, sorter_name)]
                similarities, errors = comp.compute_connectivity_by_similarity(similarity_matrix)

                all_similarities.append(similarities)
                all_errors.append(errors)
            
            all_similarities = np.concatenate(all_similarities, axis=0)
            all_errors = np.concatenate(all_errors, axis=0)
            
            idx = (all_similarities >= self.similarity_range[0]) & (all_similarities <= self.similarity_range[1])
            all_similarities = all_similarities[idx]
            all_errors = all_errors[idx]

            order = np.argsort(all_similarities)
            all_similarities = all_similarities[order]
            all_errors = all_errors[order, :]

            mean_error = np.mean(all_errors, axis=0)
            mean_error = np.nan_to_num(mean_error)
            self.ax.plot(comp.time_axis, mean_error, color='C%d' %sorter_ind, label=sorter_name)
            
            
        self.ax.set_ylabel('cc error')
        self.ax.set_xlabel('lags (ms)')

        if self.show_legend:
            self.ax.legend()


def plot_study_comparison_connectivity_by_similarity(*args, **kwargs):
    W = StudyComparisonConnectivityBySimilarityWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_connectivity_by_similarity.__doc__ = StudyComparisonConnectivityBySimilarityWidget.__doc__

def plot_study_comparison_connectivity_by_similarity_range(*args, **kwargs):
    W = StudyComparisonConnectivityBySimilarityRangeWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_connectivity_by_similarity_range.__doc__ = StudyComparisonConnectivityBySimilarityRangeWidget.__doc__


def plot_study_comparison_connectivity_by_similarity_ranges_mean_error(*args, **kwargs):
    W = StudyComparisonConnectivityBySimilarityRangesMeanErrorWidget(*args, **kwargs)
    W.plot()
    return W
plot_study_comparison_connectivity_by_similarity_ranges_mean_error.__doc__ = StudyComparisonConnectivityBySimilarityRangesMeanErrorWidget.__doc__