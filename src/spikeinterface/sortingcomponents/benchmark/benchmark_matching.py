from __future__ import annotations

from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.postprocessing import compute_template_similarity
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface import load_extractor
from spikeinterface.comparison import CollisionGTComparison, compare_sorter_to_ground_truth
from spikeinterface.widgets import (
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
    plot_unit_waveforms,
)

import time
import os
import pickle
from pathlib import Path
import string, random
import pylab as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shutil
import copy
from tqdm.auto import tqdm
from .benchmark_tools import BenchmarkStudy, Benchmark
from spikeinterface.core.basesorting import minimum_spike_dtype


class MatchingBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.method = params['method']
        self.templates = params["method_kwargs"]['templates']
        self.method_kwargs = params['method_kwargs']
    
    def run(self, **job_kwargs):
        spikes = find_spikes_from_templates(
            self.recording, 
            method=self.method, 
            method_kwargs=self.method_kwargs, 
            **job_kwargs
        )
        unit_ids = self.templates.unit_ids
        sorting = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        sorting["sample_index"] = spikes["sample_index"]
        sorting["unit_index"] = spikes["cluster_index"]
        sorting["segment_index"] = spikes["segment_index"]
        sorting = NumpySorting(sorting, self.recording.sampling_frequency, unit_ids)
        result = {'sorting' : sorting}

        ## Add metrics
        
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=True)
        result['gt_comparison'] = comp
        result['templates'] = self.templates
        result['gt_collision'] = CollisionGTComparison(self.gt_sorting, sorting, exhaustive_gt=True)
        return result
    
    def save_to_folder(self, folder, result):
        result['sorting'].save(folder = folder / "sorting", format="numpy_folder")
        result['templates'].to_zarr(folder / "templates")
        comparison_file = folder / "gt_comparison.pickle"
        with open(comparison_file, mode="wb") as f:
            pickle.dump(result['gt_comparison'], f)
    
    @classmethod
    def load_folder(cls, folder):
        result = {}
        result['sorting'] = load_extractor(folder / "sorting")
        result['templates'] = Templates.from_zarr(folder / "templates")
        with open(folder / "gt_comparison.pickle", "rb") as f:
            result['gt_comparison'] = pickle.load(f)
        return result

class MatchingStudy(BenchmarkStudy):

    benchmark_class = MatchingBenchmark

    def _run(self, keys, **job_kwargs):
        for key in keys:
            
            dataset_key = self.cases[key]["dataset"]
            recording, gt_sorting = self.datasets[dataset_key]
            params = self.cases[key]["params"]
            benchmark = MatchingBenchmark(recording, gt_sorting, params)
            result = benchmark.run()
            self.results[key] = result
            benchmark.save_to_folder(self.folder / "results" / self.key_to_str(key), result)


    def plot_agreements(self, case_keys=None, figsize=(15,15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize)

        for count, key in enumerate(case_keys):
            ax = axs[count]
            ax.set_title(self.cases[key]['label'])
            plot_agreement_matrix(self.results[key]['gt_comparison'], ax=ax)
    
    def plot_performances_vs_snr(self, case_keys=None, figsize=(15,15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=figsize)

        for count, k in enumerate(("accuracy", "recall", "precision")):
            
            ax = axs[count]
            for key in case_keys:
                label = self.cases[key]["label"]
                
                analyzer = self.get_sorting_analyzer(key)
                metrics = analyzer.get_extension('quality_metrics').get_data()
                x = metrics["snr"].values
                y = self.results[key]['gt_comparison'].get_performance()[k].values
                ax.scatter(x, y, marker=".", label=label)

            if count == 2:
                ax.legend()

    def plot_collisions(self, case_keys=None, figsize=(15,15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())
        
        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize)

        for count, key in enumerate(case_keys):
            templates_array = self.results[key]['templates'].templates_array
            plot_comparison_collision_by_similarity(
                self.results[key]['gt_collision'], templates_array, ax=axs[count], 
                show_legend=True, mode="lines", good_only=False
            )

    # def plot_errors_matching(benchmark, comp, unit_id, nb_spikes=200, metric="cosine"):
    #     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))

    #     benchmark.we.sorting.get_unit_spike_train(unit_id)
    #     template = benchmark.we.get_template(unit_id)
    #     a = template.reshape(template.size, 1).T
    #     count = 0
    #     colors = ["r", "b"]
    #     for label in ["TP", "FN"]:
    #         seg_num = 0  # TODO: make compatible with multiple segments
    #         idx_1 = np.where(comp.get_labels1(unit_id)[seg_num] == label)
    #         idx_2 = benchmark.we.get_sampled_indices(unit_id)["spike_index"]
    #         intersection = np.where(np.isin(idx_2, idx_1))[0]
    #         intersection = np.random.permutation(intersection)[:nb_spikes]
    #         if len(intersection) == 0:
    #             print(f"No {label}s found for unit {unit_id}")
    #             continue
    #         ### Should be able to give a subset of waveforms only...
    #         ax = axs[count, 0]
    #         plot_unit_waveforms(
    #             benchmark.we,
    #             unit_ids=[unit_id],
    #             axes=[ax],
    #             unit_selected_waveforms={unit_id: intersection},
    #             unit_colors={unit_id: colors[count]},
    #         )
    #         ax.set_title(label)

    #         wfs = benchmark.we.get_waveforms(unit_id)
    #         wfs = wfs[intersection, :, :]

    #         import sklearn

    #         nb_spikes = len(wfs)
    #         b = wfs.reshape(nb_spikes, -1)
    #         distances = sklearn.metrics.pairwise_distances(a, b, metric).flatten()
    #         ax = axs[count, 1]
    #         ax.set_title(label)
    #         ax.hist(distances, color=colors[count])
    #         ax.set_ylabel("# waveforms")
    #         ax.set_xlabel(metric)

    #         count += 1
    #     return fig, axs


    # def plot_errors_matching_all_neurons(benchmark, comp, nb_spikes=200, metric="cosine"):
    #     templates = benchmark.templates
    #     nb_units = len(benchmark.we.unit_ids)
    #     colors = ["r", "b"]

    #     results = {"TP": {"mean": [], "std": []}, "FN": {"mean": [], "std": []}}

    #     for i in range(nb_units):
    #         unit_id = benchmark.we.unit_ids[i]
    #         idx_2 = benchmark.we.get_sampled_indices(unit_id)["spike_index"]
    #         wfs = benchmark.we.get_waveforms(unit_id)
    #         template = benchmark.we.get_template(unit_id)
    #         a = template.reshape(template.size, 1).T

    #         for label in ["TP", "FN"]:
    #             idx_1 = np.where(comp.get_labels1(unit_id) == label)[0]
    #             intersection = np.where(np.isin(idx_2, idx_1))[0]
    #             intersection = np.random.permutation(intersection)[:nb_spikes]
    #             wfs_sliced = wfs[intersection, :, :]

    #             import sklearn

    #             all_spikes = len(wfs_sliced)
    #             if all_spikes > 0:
    #                 b = wfs_sliced.reshape(all_spikes, -1)
    #                 if metric == "cosine":
    #                     distances = sklearn.metrics.pairwise.cosine_similarity(a, b).flatten()
    #                 else:
    #                     distances = sklearn.metrics.pairwise_distances(a, b, metric).flatten()
    #                 results[label]["mean"] += [np.nanmean(distances)]
    #                 results[label]["std"] += [np.nanstd(distances)]
    #             else:
    #                 results[label]["mean"] += [0]
    #                 results[label]["std"] += [0]

    #     fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
    #     for count, label in enumerate(["TP", "FN"]):
    #         ax = axs[count]
    #         idx = np.argsort(benchmark.metrics.snr)
    #         means = np.array(results[label]["mean"])[idx]
    #         stds = np.array(results[label]["std"])[idx]
    #         ax.errorbar(benchmark.metrics.snr[idx], means, yerr=stds, c=colors[count])
    #         ax.set_title(label)
    #         ax.set_xlabel("snr")
    #         ax.set_ylabel(metric)
    #     return fig, axs

    def plot_comparison_matching(
        benchmark,
        comp_per_method,
        performance_names=["accuracy", "recall", "precision"],
        colors=["g", "b", "r"],
        ylim=(-0.1, 1.1),
    ):
        num_methods = len(benchmark.methods)
        fig, axs = plt.subplots(ncols=num_methods, nrows=num_methods, figsize=(10, 10))
        for i, method1 in enumerate(benchmark.methods):
            for j, method2 in enumerate(benchmark.methods):
                if len(axs.shape) > 1:
                    ax = axs[i, j]
                else:
                    ax = axs[j]
                comp1, comp2 = comp_per_method[method1], comp_per_method[method2]
                if i <= j:
                    for performance, color in zip(performance_names, colors):
                        perf1 = comp1.get_performance()[performance]
                        perf2 = comp2.get_performance()[performance]
                        ax.plot(perf2, perf1, ".", label=performance, color=color)

                    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
                    ax.set_ylim(ylim)
                    ax.set_xlim(ylim)
                    ax.spines[["right", "top"]].set_visible(False)
                    ax.set_aspect("equal")

                    if j == i:
                        ax.set_ylabel(f"{method1}")
                    else:
                        ax.set_yticks([])
                    if i == j:
                        ax.set_xlabel(f"{method2}")
                    else:
                        ax.set_xticks([])
                    if i == num_methods - 1 and j == num_methods - 1:
                        patches = []
                        for color, name in zip(colors, performance_names):
                            patches.append(mpatches.Patch(color=color, label=name))
                        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
                else:
                    ax.spines["bottom"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
        plt.tight_layout(h_pad=0, w_pad=0)
        return fig, axs