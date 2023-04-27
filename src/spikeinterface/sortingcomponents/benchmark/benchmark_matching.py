
from spikeinterface.core import extract_waveforms
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.postprocessing import compute_template_similarity
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import CollisionGTComparison, compare_sorter_to_ground_truth
from spikeinterface.widgets import (plot_sorting_performance,
    plot_agreement_matrix, plot_comparison_collision_by_similarity,
    plot_unit_templates, plot_unit_waveforms, plot_gt_performances)


import time
import os
import string, random
import pylab as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shutil

class BenchmarkMatching:

    def __init__(self, recording, gt_sorting, waveform_extractor, methods, methods_kwargs=None, exhaustive_gt=True,
                 tmp_folder=None, **job_kwargs):
        self.methods = methods
        if methods_kwargs is None:
            methods_kwargs = {method:{} for method in methods}
        self.methods_kwargs = methods_kwargs
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.exhaustive_gt = exhaustive_gt
        self.sampling_rate = self.recording.get_sampling_frequency()

        if tmp_folder is None:
            tmp_folder = os.path.join('.', ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)))
        self.tmp_folder = tmp_folder

        self.we = waveform_extractor
        for method in self.methods:
            self.methods_kwargs[method]['waveform_extractor'] = self.we
        self.templates = self.we.get_all_templates(mode='median')
        self.metrics = compute_quality_metrics(self.we, metric_names=['snr'], load_if_exists=True)
        self.similarity = compute_template_similarity(self.we)


    def run_matching(self, methods_kwargs, collision=False):
        comps = {}
        for method in self.methods:
            spikes = find_spikes_from_templates(self.recording, method=method,
                                                method_kwargs=methods_kwargs[method],
                                                **self.job_kwargs)
            sorting = NumpySorting.from_times_labels(spikes['sample_index'], spikes['cluster_index'],
                                                     self.sampling_rate)
            if collision:
                comp = CollisionGTComparison(self.gt_sorting, sorting, exhaustive_gt=self.exhaustive_gt)
            else:
                comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting)
            comps[method] = comp
        return comps


    def run_matching_num_spikes(self, spike_num, seed=0, we_kwargs=None, template_mode='median'):
        if we_kwargs is None:
            we_kwargs = {}
        we_kwargs.update(dict(max_spikes_per_unit=spike_num, seed=seed, overwrite=True, load_if_exists=False,
                              **self.job_kwargs))
        np.random.seed(seed)

        # Generate New Waveform Extractor with New Spike Numbers
        we = extract_waveforms(self.recording, self.gt_sorting, self.tmp_folder, **we_kwargs)
        methods_kwargs = self.update_methods_kwargs(we, template_mode)

        comps = self.run_matching(methods_kwargs)
        shutil.rmtree(self.tmp_folder)
        return comps

    def update_methods_kwargs(self, we, template_mode='median'):
        templates = we.get_all_templates(we.unit_ids, mode=template_mode)
        methods_kwargs = self.methods_kwargs.copy()
        for method in self.methods:
            method_kwargs = methods_kwargs[method]
            if method == 'wobble':
                method_kwargs.update(dict(templates=templates, nbefore=we.nbefore, nafter=we.nafter))
            else:
                method_kwargs['waveform_extractor'] = we
        return methods_kwargs


    def run_matching_misclassed(self, fraction_misclassed, min_similarity=-1, seed=0, we_kwargs=None,
                                template_mode='median'):
        if we_kwargs is None:
            we_kwargs = {}
        we_kwargs.update(dict(seed=seed, overwrite=True, load_if_exists=False, **self.job_kwargs))
        np.random.seed(seed)

        # Randomly misclass spike trains
        spike_time_indices, labels = [], []
        for unit_index, unit_id in enumerate(self.we.unit_ids):
            unit_sorting = self.gt_sorting.get_unit_spike_train(unit_id=unit_id)
            unit_similarity = self.similarity[unit_index, :]
            unit_similarity[unit_index] = min_similarity - 1  # skip self
            similar_unit_ids = self.we.unit_ids[unit_similarity >= min_similarity]
            at_least_one_similar_unit = len(similar_unit_ids)
            num_spikes = int(len(unit_sorting) * fraction_misclassed)

            unit_misclass_idx = np.random.choice(np.arange(len(unit_sorting)), size=num_spikes, replace=False)
            for i, spike in enumerate(unit_sorting):
                spike_time_indices.append(spike)
                if i in unit_misclass_idx and at_least_one_similar_unit:
                    alt_id = np.random.choice(similar_unit_ids)
                    labels.append(alt_id)
                else:
                    labels.append(unit_id)
        spike_time_indices = np.array(spike_time_indices)
        labels = np.array(labels)
        sort_idx = np.argsort(spike_time_indices)
        spike_time_indices = spike_time_indices[sort_idx]
        labels = labels[sort_idx]
        sorting_misclassed = NumpySorting.from_times_labels(spike_time_indices, labels, self.sampling_rate)

        # Generate New Waveform Extractor with Misclassed Spike Trains
        we = extract_waveforms(self.recording, sorting_misclassed, self.tmp_folder, **we_kwargs)
        methods_kwargs = self.update_methods_kwargs(we, template_mode)

        comps = self.run_matching(methods_kwargs)
        shutil.rmtree(self.tmp_folder)
        return comps


    def run_matching_missing_units(self, fraction_missing, snr_threshold=0, seed=0, we_kwargs=None, template_mode='median'):
        if we_kwargs is None:
            we_kwargs = {}
        we_kwargs.update(dict(seed=seed, overwrite=True, load_if_exists=False, **self.job_kwargs))
        np.random.seed(seed)

        # Omit fraction_missing of units with lowest SNR
        metrics = self.metrics.sort_values('snr')
        missing_units = np.array(metrics.index[metrics.snr < snr_threshold])
        num_missing = int(len(missing_units) * fraction_missing)
        missing_units = np.random.choice(missing_units, size=num_missing, replace=False)
        present_units = np.setdiff1d(self.we.unit_ids, missing_units)
        spike_time_indices, spike_cluster_ids = [], []
        for unit in present_units:
            spike_train = self.gt_sorting.get_unit_spike_train(unit)
            for time_index in spike_train:
                spike_time_indices.append(time_index)
                spike_cluster_ids.append(unit)
        spike_time_indices = np.array(spike_time_indices)
        spike_cluster_ids = np.array(spike_cluster_ids)
        sorting = NumpySorting.from_times_labels(spike_time_indices, spike_cluster_ids, self.sampling_rate)

        # Generate New Waveform Extractor with Missing Units
        we = extract_waveforms(self.recording, sorting, self.tmp_folder, **we_kwargs)
        methods_kwargs = self.update_methods_kwargs(we, template_mode)

        comps = self.run_matching(methods_kwargs)
        shutil.rmtree(self.tmp_folder)
        return comps


    def run_matching_vary_parameter(self, parameters, parameter_name, num_replicates=1, we_kwargs=None,
                                    template_mode='median', verbose=False, **kwargs):
        if parameter_name == 'num_spikes':
            run_matching_fn = self.run_matching_num_spikes
        elif parameter_name == 'fraction_misclassed':
            run_matching_fn = self.run_matching_misclassed
        elif parameter_name == 'fraction_missing':
            run_matching_fn = self.run_matching_missing_units
        else:
            raise ValueError(
                "parameter_name must be one of ['num_spikes', 'fraction_misclassed', 'snr_threshold'],")
        comps, parameter_values, parameter_names, iter_nums, methods = [], [], [], [], []
        for parameter in parameters:
            if verbose:
                print(f"{parameter_name} = {parameter}")
            for i in range(1, num_replicates+1):
                if verbose:
                    print(f"{i = }")
                comp_per_method = run_matching_fn(parameter, seed=i, we_kwargs=we_kwargs, template_mode=template_mode,
                                                  **kwargs)
                for method in self.methods:
                    comps.append(comp_per_method[method])
                    parameter_values.append(parameter)
                    parameter_names.append(parameter_name)
                    iter_nums.append(i)
                    methods.append(method)
        comparisons = pd.DataFrame({'comp': comps,
                                    'parameter_value': parameter_values,
                                    'parameter_name' : parameter_names,
                                    'iter_num': iter_nums,
                                    'method': methods})
        return comparisons


    def plot(self, comp, title=None):
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = axs[0, 0]
        ax.set_title(title)
        plot_agreement_matrix(comp, ax=ax)
        ax.set_title(title)
        
        ax = axs[1, 0]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_sorting_performance(comp, self.metrics, performance_name='accuracy', metric_name='snr', ax=ax, color='r')
        plot_sorting_performance(comp, self.metrics, performance_name='recall', metric_name='snr', ax=ax, color='g')
        plot_sorting_performance(comp, self.metrics, performance_name='precision', metric_name='snr', ax=ax, color='b')
        ax.legend(['accuracy', 'recall', 'precision'])
        
        ax = axs[1, 1]
        plot_gt_performances(comp, ax=ax)

        ax = axs[0, 1]
        if self.exhaustive_gt:
            plot_comparison_collision_by_similarity(comp, self.templates, ax=ax, show_legend=True, mode='lines',
                                                    good_only=False)
        return fig, axs

def plot_errors_matching(benchmark, comp, unit_id, nb_spikes=200, metric='cosine'):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    
    benchmark.we.sorting.get_unit_spike_train(unit_id)
    template = benchmark.we.get_template(unit_id)
    a = template.reshape(template.size, 1).T
    count = 0
    colors = ['r', 'b']
    for label in ['TP', 'FN']:
        seg_num = 0 # TODO: make compatible with multiple segments
        idx_1 = np.where(comp.get_labels1(unit_id)[seg_num] == label)
        idx_2 = benchmark.we.get_sampled_indices(unit_id)['spike_index']
        intersection = np.where(np.in1d(idx_2, idx_1))[0]
        intersection = np.random.permutation(intersection)[:nb_spikes]
        if len(intersection) == 0:
            print(f"No {label}s found for unit {unit_id}")
            continue
        ### Should be able to give a subset of waveforms only...
        ax = axs[count, 0]
        plot_unit_waveforms(benchmark.we, unit_ids=[unit_id], axes=[ax],
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

def plot_errors_matching_all_neurons(benchmark, comp, nb_spikes=200, metric='cosine'):
    templates = benchmark.templates
    nb_units = len(benchmark.we.unit_ids)
    colors = ['r', 'b']

    results = {'TP' : {'mean' : [], 'std' : []}, 
               'FN' : {'mean' : [], 'std' : []}}
    
    for i in range(nb_units):
        unit_id = benchmark.we.unit_ids[i]
        idx_2 = benchmark.we.get_sampled_indices(unit_id)['spike_index']
        wfs = benchmark.we.get_waveforms(unit_id)
        template = benchmark.we.get_template(unit_id)
        a = template.reshape(template.size, 1).T
        
        for label in ['TP', 'FN']:
            idx_1 = np.where(comp.get_labels1(unit_id) == label)[0]
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


def plot_comparison_matching(benchmark, comp_per_method, performance_names=['accuracy', 'recall', 'precision'],
                             colors = ['g', 'b', 'r'], ylim=(-0.1, 1.1)):
    num_methods = len(benchmark.methods)
    fig, axs = plt.subplots(ncols=num_methods, nrows=num_methods, figsize=(10, 10))
    for i, method1 in enumerate(benchmark.methods):
        for j, method2 in enumerate(benchmark.methods):
            if len(axs.shape) > 1:
                ax = axs[i, j]
            else:
                ax = axs[j]
            comp1, comp2 = comp_per_method[method1], comp_per_method[method2]
            for performance, color in zip(performance_names, colors):
                perf1 = comp1.get_performance()[performance]
                perf2 = comp2.get_performance()[performance]
                ax.plot(perf2, perf1, '.', label=performance, color=color)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_ylim(ylim)
            ax.set_xlim(ylim)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_aspect('equal')

            if j == 0:
                ax.set_ylabel(f'{method1}')
            else:
                ax.set_yticks([])
            if i == num_methods - 1:
                ax.set_xlabel(f'{method2}')
            else:
                ax.set_xticks([])
            if i == num_methods - 1 and j == num_methods - 1:
                patches = []
                for color, name in zip(colors, performance_names):
                    patches.append(mpatches.Patch(color=color, label=name))
                ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(h_pad=0, w_pad=0)
