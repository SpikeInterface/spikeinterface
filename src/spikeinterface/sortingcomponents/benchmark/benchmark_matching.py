from __future__ import annotations

from spikeinterface.postprocessing import compute_template_similarity
from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core.template import Templates
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import CollisionGTComparison, compare_sorter_to_ground_truth
from spikeinterface.widgets import (
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
)

from pathlib import Path
import pylab as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from .benchmark_tools import BenchmarkStudy, Benchmark
from spikeinterface.core.basesorting import minimum_spike_dtype


class MatchingBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.method = params["method"]
        self.templates = params["method_kwargs"]["templates"]
        self.method_kwargs = params["method_kwargs"]
        self.result = {}

    def run(self, **job_kwargs):
        spikes = find_spikes_from_templates(
            self.recording, method=self.method, method_kwargs=self.method_kwargs, **job_kwargs
        )
        unit_ids = self.templates.unit_ids
        sorting = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        sorting["sample_index"] = spikes["sample_index"]
        sorting["unit_index"] = spikes["cluster_index"]
        sorting["segment_index"] = spikes["segment_index"]
        sorting = NumpySorting(sorting, self.recording.sampling_frequency, unit_ids)
        self.result = {"sorting": sorting}
        self.result["templates"] = self.templates

    def compute_result(self, **result_params):
        sorting = self.result["sorting"]
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=True)
        self.result["gt_comparison"] = comp
        self.result["gt_collision"] = CollisionGTComparison(self.gt_sorting, sorting, exhaustive_gt=True)

    _run_key_saved = [
        ("sorting", "sorting"),
        ("templates", "zarr_templates"),
    ]
    _result_key_saved = [("gt_collision", "pickle"), ("gt_comparison", "pickle")]


class MatchingStudy(BenchmarkStudy):

    benchmark_class = MatchingBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        benchmark = MatchingBenchmark(recording, gt_sorting, params)
        return benchmark

    def plot_agreements(self, case_keys=None, figsize=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            ax.set_title(self.cases[key]["label"])
            plot_agreement_matrix(self.get_result(key)["gt_comparison"], ax=ax)

    def plot_performances_vs_snr(self, case_keys=None, figsize=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=figsize)

        for count, k in enumerate(("accuracy", "recall", "precision")):

            ax = axs[count]
            for key in case_keys:
                label = self.cases[key]["label"]

                analyzer = self.get_sorting_analyzer(key)
                metrics = analyzer.get_extension("quality_metrics").get_data()
                x = metrics["snr"].values
                y = self.get_result(key)["gt_comparison"].get_performance()[k].values
                ax.scatter(x, y, marker=".", label=label)
                ax.set_title(k)

            if count == 2:
                ax.legend()

    def plot_collisions(self, case_keys=None, figsize=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            templates_array = self.get_result(key)["templates"].templates_array
            plot_comparison_collision_by_similarity(
                self.get_result(key)["gt_collision"],
                templates_array,
                ax=axs[0, count],
                show_legend=True,
                mode="lines",
                good_only=False,
            )

    def plot_comparison_matching(
        self,
        case_keys=None,
        performance_names=["accuracy", "recall", "precision"],
        colors=["g", "b", "r"],
        ylim=(-0.1, 1.1),
        figsize=None,
    ):

        if case_keys is None:
            case_keys = list(self.cases.keys())

        num_methods = len(case_keys)
        fig, axs = plt.subplots(ncols=num_methods, nrows=num_methods, figsize=(10, 10))
        for i, key1 in enumerate(case_keys):
            for j, key2 in enumerate(case_keys):
                if len(axs.shape) > 1:
                    ax = axs[i, j]
                else:
                    ax = axs[j]
                comp1 = self.get_result(key1)["gt_comparison"]
                comp2 = self.get_result(key2)["gt_comparison"]
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

                    label1 = self.cases[key1]["label"]
                    label2 = self.cases[key2]["label"]
                    if j == i:
                        ax.set_ylabel(f"{label1}")
                    else:
                        ax.set_yticks([])
                    if i == j:
                        ax.set_xlabel(f"{label2}")
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
