
from spikeinterface.core import extract_waveforms
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import CollisionGTComparison
from spikeinterface.widgets import (plot_sorting_performance,
    plot_agreement_matrix, plot_comparison_collision_by_similarity,
    plot_unit_templates, plot_unit_waveforms, plot_gt_performances)


import time
import os
import string, random
import pylab as plt
import numpy as np

class BenchmarkMatching:

    def __init__(self, recording, gt_sorting, method, exhaustive_gt=True, method_kwargs={}, tmp_folder=None, job_kwargs={}):
        self.method = method
        self.method_kwargs = method_kwargs
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.exhaustive_gt = exhaustive_gt
        self.recording_f = bandpass_filter(self.recording,  dtype='float32')
        self.recording_f = common_reference(self.recording_f)
        self.sampling_rate = self.recording_f.get_sampling_frequency()
        self.job_kwargs = job_kwargs
        self.method_kwargs = method_kwargs
        self.metrics = None

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join('.', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))

        self.we = extract_waveforms(self.recording_f, self.gt_sorting, self.tmp_folder, load_if_exists=True,
                                   ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500,
                                   **self.job_kwargs)

        self.method_kwargs.update({'waveform_extractor' : self.we})
        self.templates = self.we.get_all_templates(mode='median')
   
    def __del__(self):
        import shutil
        shutil.rmtree(self.tmp_folder)

    def run(self):
        t_start = time.time()
        spikes = find_spikes_from_templates(self.recording_f, method=self.method, method_kwargs=self.method_kwargs, **self.job_kwargs)
        self.run_time = time.time() - t_start
        sorting = NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], self.sampling_rate)
        self.comp = CollisionGTComparison(self.gt_sorting, sorting, exhaustive_gt=self.exhaustive_gt)
        self.metrics = compute_quality_metrics(self.we, metric_names=['snr'], load_if_exists=True)

    def plot(self, title=None):
        
        if title is None:
            title = self.method

        if self.metrics is None:
            self.metrics = compute_quality_metrics(self.we, metric_names=['snr'], load_if_exists=True)

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = axs[0, 0]
        ax.set_title(title)
        plot_agreement_matrix(self.comp, ax=ax)
        ax.set_title(title)
        
        ax = axs[1, 0]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_sorting_performance(self.comp, self.metrics, performance_name='accuracy', metric_name='snr', ax=ax, color='r')
        plot_sorting_performance(self.comp, self.metrics, performance_name='recall', metric_name='snr', ax=ax, color='g')
        plot_sorting_performance(self.comp, self.metrics, performance_name='precision', metric_name='snr', ax=ax, color='b')        
        #ax.set_ylim(0.8, 1)
        ax.legend(['accuracy', 'recall', 'precision'])
        
        ax = axs[1, 1]
        plot_gt_performances(self.comp, ax=ax)

        ax = axs[0, 1]
        if self.exhaustive_gt:
            plot_comparison_collision_by_similarity(self.comp, self.templates, ax=ax, show_legend=True, mode='lines')

def plot_errors_matching(benchmark, unit_id, nb_spikes=200, metric='cosine'):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    
    benchmark.we.sorting.get_unit_spike_train(unit_id)
    template = benchmark.we.get_template(unit_id)
    a = template.reshape(template.size, 1).T
    count = 0
    colors = ['r', 'b']
    for label in ['TP', 'FN']:
        idx_1 = np.where(benchmark.comp.get_labels1(unit_id) == label)
        idx_2 = benchmark.we.get_sampled_indices(unit_id)['spike_index']
        intersection = np.where(np.in1d(idx_2, idx_1))[0]
        intersection = np.random.permutation(intersection)[:nb_spikes]
        ### Should be able to give a subset of waveforms only...
        ax = axs[count, 0]
        plot_unit_waveforms(benchmark.we, unit_ids=[unit_id], axes=ax, 
                            unit_selected_waveforms = {unit_id : intersection},
                            unit_colors = {unit_id : colors[count]})   
        ax.set_title(label)
        
        wfs = benchmark.we.get_waveforms(unit_id)
        wfs = wfs[intersection, :, :]
                
        import sklearn
        
        nb_spikes = len(wfs)
        b = wfs.reshape(nb_spikes, -1)
        distances = sklearn.metrics.pairwise_distances(a, b, metric).flatten()
        ax = axs[count, 1]
        ax.set_title(label)
        ax.hist(distances, color=colors[count])
        ax.set_ylabel('# waveforms')
        ax.set_xlabel(metric)
        
        count += 1

def plot_errors_matching_all_neurons(benchmark, nb_spikes=200, metric='cosine'):
    templates = benchmark.templates
    nb_units = len(benchmark.we.sorting.unit_ids)
    colors = ['r', 'b']

    results = {'TP' : {'mean' : [], 'std' : []}, 
               'FN' : {'mean' : [], 'std' : []}}
    
    for i in range(nb_units):
        unit_id = benchmark.we.sorting.unit_ids[i]
        idx_2 = benchmark.we.get_sampled_indices(unit_id)['spike_index']
        wfs = benchmark.we.get_waveforms(unit_id)
        template = benchmark.we.get_template(unit_id)
        a = template.reshape(template.size, 1).T
        
        for label in ['TP', 'FN']:
            idx_1 = np.where(benchmark.comp.get_labels1(unit_id) == label)[0] 
            intersection = np.where(np.in1d(idx_2, idx_1))[0]
            intersection = np.random.permutation(intersection)[:nb_spikes]            
            wfs_sliced = wfs[intersection, :, :]
                    
            import sklearn
            
            all_spikes = len(wfs_sliced)
            if all_spikes > 0:
                b = wfs_sliced.reshape(all_spikes, -1)
                if metric == 'cosine':
                    distances = sklearn.metrics.pairwise.cosine_similarity(a, b).flatten()
                else:
                    distances = sklearn.metrics.pairwise_distances(a, b, metric).flatten()
                results[label]['mean'] += [np.nanmean(distances)]
                results[label]['std'] += [np.nanstd(distances)]
            else:
                results[label]['mean'] += [0]
                results[label]['std'] += [0]
    
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
    for count, label in enumerate(['TP', 'FN']):
        ax = axs[count]
        idx = np.argsort(benchmark.metrics.snr)
        means = np.array(results[label]['mean'])[idx]
        stds = np.array(results[label]['std'])[idx]
        ax.errorbar(benchmark.metrics.snr[idx], means, yerr=stds, c=colors[count])
        ax.set_title(label)
        ax.set_xlabel('snr')
        ax.set_ylabel(metric)


def plot_comparison_matching(benchmarks, performance_names=['accuracy', 'recall', 'precision'], colors = ['g', 'b', 'r'], ylim=(0.5, 1)):
    nb_benchmarks = len(benchmarks)
    fig, axs = plt.subplots(ncols=nb_benchmarks, nrows=nb_benchmarks - 1, figsize=(10, 10))
    for i in range(nb_benchmarks - 1):
        for j in range(nb_benchmarks):
            ax = axs[i, j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if j > i:
                for performance, color in zip(performance_names, colors):
                    perf1 = benchmarks[i].comp.get_performance()[performance]
                    perf2 = benchmarks[j].comp.get_performance()[performance]
                    ax.plot(perf2, perf1, '.', label=performance, color=color)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.set_ylim(ylim)
                ax.set_xlim(ylim)
                
                if j > i + 1:
                    ax.set_yticks([], [])
                    ax.set_xticks([], [])
            
                if j == i + 1:
                    ax.set_ylabel(f'{benchmarks[i].method}')
                    ax.set_xlabel(f'{benchmarks[j].method}')
            else:                    
                ax.set_yticks([], [])
                ax.set_xticks([], [])
                if (i == 0) and (j == 0):
                    for color, k in zip(colors, range(len(performance_names))):
                        ax.plot([0,0],[0,0],c=color)
                    ax.legend(performance_names)
    plt.tight_layout()

    # def plot_average_comparison_matching(benchmarks, performance_names=['recall'], ylim=(0, 1)):    
    #     nb_benchmarks = len(benchmarks)
    #     results = {}
    #     for i in range(nb_benchmarks):
    #         results[benchmarks[i].method] = {}
    #         for performance in performance_names:
    #             results[benchmarks[i].method][performance] = benchmarks[i].comp.get_performance()[performance]
        
    #     width = 1/(ncol+2)
    #     methods = [i.method for i in benchmarks]
        
    #     for c, col in enumerate(methods):
    #         x = np.arange(performances) + 1 + c / (ncol + 2)
    #         yerr = results[]
    #         ax.bar(x, m[col].values.flatten(), yerr=yerr.flatten(), width=width, color=cmap(c), label=clean_labels[c])

    #     ax.legend()

    #     ax.set_xticks(np.arange(sorter_names.size) + 1 + width)
    #     ax.set_xticklabels(sorter_names, rotation=45)
    #     ax.set_ylabel('metric')
    #     ax.set_xlim(0, sorter_names.size + 1)
    #     