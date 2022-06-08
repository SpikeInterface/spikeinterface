
from spikeinterface.core import extract_waveforms
from spikeinterface.toolkit import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.extractors import read_mearec
from spikeinterface.core import NumpySorting
from spikeinterface.toolkit.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import CollisionGTComparison
from spikeinterface.widgets import plot_sorting_performance, plot_agreement_matrix, plot_comparison_collision_by_similarity

import time
import string, random
import pylab as plt

class BenchmarkMatching:

    def __init__(self, mearec_file, method, method_kwargs={}, tmp_folder=None, job_kwargs={}):
        self.mearec_file = mearec_file
        self.method = method
        self.method_kwargs = method_kwargs
        self.recording, self.gt_sorting = read_mearec(mearec_file)
        self.recording_f = bandpass_filter(self.recording, dtype='float32')
        self.recording_f = common_reference(self.recording_f)
        self.sampling_rate = self.recording_f.get_sampling_frequency()
        self.job_kwargs = job_kwargs
        self.method_kwargs = method_kwargs
        self.metrics = None

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = ''.join('waveforms')

        self.we = extract_waveforms(self.recording_f, self.gt_sorting, self.tmp_folder, load_if_exists=True,
                                   ms_before=2.5, ms_after=3.5, max_spikes_per_unit=500,
                                   **self.job_kwargs)

        self.method_kwargs.update({'waveform_extractor' : self.we})
        self.templates = self.we.get_all_templates(mode='median')
   
    def run(self):
        t_start = time.time()
        spikes = find_spikes_from_templates(self.recording_f, method=self.method, method_kwargs=self.method_kwargs, **self.job_kwargs)
        self.run_time = time.time() - t_start
        sorting = NumpySorting.from_times_labels(spikes['sample_ind'], spikes['cluster_ind'], self.sampling_rate)
        self.comp = CollisionGTComparison(self.gt_sorting, sorting)
    
    def compute_benchmark(self, metrics_names=['snr']):
        self.metrics = compute_quality_metrics(self.we, metric_names=metrics_names, load_if_exists=True)

    def plot(self, title=None):
        
        if title is None:
            title = self.method

        if self.metrics is None:
            self.compute_benchmark()

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = axs[0, 0]
        ax.set_title(title)
        plot_agreement_matrix(self.comp, ax=ax)
        ax.set_title(title)
        
        ax = axs[1, 0]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_sorting_performance(self.comp, self.metrics, performance_name='accuracy', metric_name='snr', ax=ax, color='g')
        plot_sorting_performance(self.comp, self.metrics, performance_name='recall', metric_name='snr', ax=ax, color='b')
        plot_sorting_performance(self.comp, self.metrics, performance_name='precision', metric_name='snr', ax=ax, color='r')        
        #ax.set_ylim(0.8, 1)
        ax.legend(['accuracy', 'recall', 'precision'])
        
        ax = axs[1, 1]

        ax = axs[0, 1]
        plot_comparison_collision_by_similarity(self.comp, self.templates, ax=ax, show_legend=True, mode='lines')

def plot_errors_matching(benchmark, unit_id):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(10, 10))
    ax = axs[0]
    from spikeinterface.widgets import plot_unit_templates, plot_unit_waveforms
    plot_unit_templates(benchmark.we, unit_ids=[unit_id], axes=ax)

    benchmark.we.sorting.get_unit_spike_train(unit_id)
    count = 1
    for label in ['TP', 'FN']:
        idx_1 = np.where(benchmark.comp.get_labels1(unit_id) == label)
        idx_2 = benchmark.we.get_sampled_indices(unit_id)['spike_index']
        intersection = np.in1d(idx_2, idx_1)
        ### SHould be able to give a subset of waveforms only...
        plot_unit_waveforms(benchmark.we, unit_ids=[unit_id], axes=axs[count], selection = {unit_id : idx})        
        count += 1

def plot_comparison_matching(benchmarks, performance_names=['recall'], ylim=(0.5, 1)):
    nb_benchmarks = len(benchmarks)
    fig, axs = plt.subplots(ncols=nb_benchmarks, nrows=nb_benchmarks - 1)
    for i in range(nb_benchmarks - 1):
        for j in range(nb_benchmarks):
            ax = axs[i, j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if j > i:
                for performance in performance_names:
                    perf1 = benchmarks[i].comp.get_performance()[performance]
                    perf2 = benchmarks[j].comp.get_performance()[performance]
                    ax.plot(perf1, perf2, '.', label=performance)
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
                    ax.plot([0,0],[0,0])
                    ax.plot([0,0],[0,0])
                    ax.legend(performance_names)
    plt.tight_layout()