from __future__ import annotations

from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import CollisionGTComparison, compare_sorter_to_ground_truth
from spikeinterface.widgets import (
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
)

import numpy as np
from .benchmark_base import Benchmark, BenchmarkStudy
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

    def compute_result(self, with_collision=False, **result_params):
        sorting = self.result["sorting"]
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=True)
        self.result["gt_comparison"] = comp
        if with_collision:
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

    def plot_agreement_matrix(self, **kwargs):
        from .benchmark_plot_tools import plot_agreement_matrix
        return plot_agreement_matrix(self, **kwargs)

    def plot_performances_vs_snr(self, **kwargs):
        from .benchmark_plot_tools import plot_performances_vs_snr
        return plot_performances_vs_snr(self, **kwargs)


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

        return fig

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
        import pylab as plt

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
                        import matplotlib.patches as mpatches

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

        return fig

    def get_count_units(self, case_keys=None, well_detected_score=None, redundant_score=None, overmerged_score=None):
        import pandas as pd

        if case_keys is None:
            case_keys = list(self.cases.keys())

        if isinstance(case_keys[0], str):
            index = pd.Index(case_keys, name=self.levels)
        else:
            index = pd.MultiIndex.from_tuples(case_keys, names=self.levels)

        columns = ["num_gt", "num_sorter", "num_well_detected"]
        comp = self.get_result(case_keys[0])["gt_comparison"]
        if comp.exhaustive_gt:
            columns.extend(["num_false_positive", "num_redundant", "num_overmerged", "num_bad"])
        count_units = pd.DataFrame(index=index, columns=columns, dtype=int)

        for key in case_keys:
            comp = self.get_result(key)["gt_comparison"]
            assert comp is not None, "You need to do study.run_comparisons() first"

            gt_sorting = comp.sorting1
            sorting = comp.sorting2

            count_units.loc[key, "num_gt"] = len(gt_sorting.get_unit_ids())
            count_units.loc[key, "num_sorter"] = len(sorting.get_unit_ids())
            count_units.loc[key, "num_well_detected"] = comp.count_well_detected_units(well_detected_score)

            if comp.exhaustive_gt:
                count_units.loc[key, "num_redundant"] = comp.count_redundant_units(redundant_score)
                count_units.loc[key, "num_overmerged"] = comp.count_overmerged_units(overmerged_score)
                count_units.loc[key, "num_false_positive"] = comp.count_false_positive_units(redundant_score)
                count_units.loc[key, "num_bad"] = comp.count_bad_units()

        return count_units

    def plot_unit_counts(self, case_keys=None, figsize=None):
        from spikeinterface.widgets.widget_list import plot_study_unit_counts

        plot_study_unit_counts(self, case_keys, figsize=figsize)

    def plot_unit_losses(self, before, after, metric=["precision"], figsize=None):

        fig, axs = plt.subplots(ncols=1, nrows=len(metric), figsize=figsize, squeeze=False)

        for count, k in enumerate(metric):

            ax = axs[0, count]

            label = self.cases[after]["label"]

            positions = self.get_result(before)["gt_comparison"].sorting1.get_property("gt_unit_locations")

            analyzer = self.get_sorting_analyzer(before)
            metrics_before = analyzer.get_extension("quality_metrics").get_data()
            x = metrics_before["snr"].values

            y_before = self.get_result(before)["gt_comparison"].get_performance()[k].values
            y_after = self.get_result(after)["gt_comparison"].get_performance()[k].values
            # if count < 2:
            # ax.set_xticks([], [])
            # elif count == 2:
            ax.set_xlabel("depth (um)")
            im = ax.scatter(positions[:, 1], x, c=(y_after - y_before), cmap="coolwarm")
            fig.colorbar(im, ax=ax, label=k)
            im.set_clim(-1, 1)
            ax.set_title(k)
            ax.set_ylabel("snr")

        # fig.subplots_adjust(right=0.85)
        # cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.75])
        # cbar = fig.colorbar(im, cax=cbar_ax, label=metric)

        # if count == 2:
        #    ax.legend()
        return fig
