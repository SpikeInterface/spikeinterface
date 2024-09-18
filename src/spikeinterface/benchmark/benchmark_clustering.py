from __future__ import annotations

from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import (
    plot_probe_map,
    plot_agreement_matrix,
)


import numpy as np

from .benchmark_base import Benchmark, BenchmarkStudy
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.template_tools import get_template_extremum_channel


class ClusteringBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, indices, peaks, exhaustive_gt=True):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.indices = indices
        self.peaks = peaks[self.indices]
        self.params = params
        self.exhaustive_gt = exhaustive_gt
        self.method = params["method"]
        self.method_kwargs = params["method_kwargs"]
        self.result = {}

    def run(self, **job_kwargs):
        labels, peak_labels = find_cluster_from_peaks(
            self.recording, self.peaks, method=self.method, method_kwargs=self.method_kwargs, **job_kwargs
        )
        self.result["peak_labels"] = peak_labels

    def compute_result(self, **result_params):
        self.noise = self.result["peak_labels"] < 0
        spikes = self.gt_sorting.to_spike_vector()
        self.result["sliced_gt_sorting"] = NumpySorting(
            spikes[self.indices], self.recording.sampling_frequency, self.gt_sorting.unit_ids
        )

        data = spikes[self.indices][~self.noise]
        # data["unit_index"] = self.result["peak_labels"][~self.noise]
        gt_unit_locations = self.gt_sorting.get_property("gt_unit_locations")
        if gt_unit_locations is None:
            print("'gt_unit_locations' is not a property of the sorting so compute it")
            gt_analyzer = create_sorting_analyzer(self.gt_sorting, self.recording, format="memory", sparse=True)
            gt_analyzer.compute(["random_spikes", "templates"])
            ext = gt_analyzer.compute("unit_locations", method="monopolar_triangulation")
            gt_unit_locations = ext.get_data()
            self.gt_sorting.set_property("gt_unit_locations", gt_unit_locations)

        self.result["sliced_gt_sorting"].set_property("gt_unit_locations", gt_unit_locations)

        self.result["clustering"] = NumpySorting.from_times_labels(
            data["sample_index"], self.result["peak_labels"][~self.noise], self.recording.sampling_frequency
        )

        self.result["gt_comparison"] = GroundTruthComparison(
            self.result["sliced_gt_sorting"], self.result["clustering"], exhaustive_gt=self.exhaustive_gt
        )

        sorting_analyzer = create_sorting_analyzer(
            self.result["sliced_gt_sorting"], self.recording, format="memory", sparse=False
        )
        sorting_analyzer.compute("random_spikes")
        ext = sorting_analyzer.compute("templates")
        self.result["sliced_gt_templates"] = ext.get_data(outputs="Templates")

        sorting_analyzer = create_sorting_analyzer(
            self.result["clustering"], self.recording, format="memory", sparse=False
        )
        sorting_analyzer.compute("random_spikes")
        ext = sorting_analyzer.compute("templates")
        self.result["clustering_templates"] = ext.get_data(outputs="Templates")

    _run_key_saved = [("peak_labels", "npy")]

    _result_key_saved = [
        ("gt_comparison", "pickle"),
        ("sliced_gt_sorting", "sorting"),
        ("clustering", "sorting"),
        ("sliced_gt_templates", "zarr_templates"),
        ("clustering_templates", "zarr_templates"),
    ]


class ClusteringStudy(BenchmarkStudy):

    benchmark_class = ClusteringBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        benchmark = ClusteringBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark

    def homogeneity_score(self, ignore_noise=True, case_keys=None):

        if case_keys is None:
            case_keys = list(self.cases.keys())

        for count, key in enumerate(case_keys):
            result = self.get_result(key)
            noise = result["peak_labels"] < 0
            from sklearn.metrics import homogeneity_score

            gt_labels = self.benchmarks[key].gt_sorting.to_spike_vector()["unit_index"]
            gt_labels = gt_labels[self.benchmarks[key].indices]
            found_labels = result["peak_labels"]
            if ignore_noise:
                gt_labels = gt_labels[~noise]
                found_labels = found_labels[~noise]
            print(
                self.cases[key]["label"],
                "Homogeneity:",
                homogeneity_score(gt_labels, found_labels),
                "Noise (%):",
                np.mean(noise),
            )

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

    # plotting by methods
    def plot_unit_counts(self, **kwargs):
        from .benchmark_plot_tools import plot_unit_counts
        return plot_unit_counts(self, **kwargs)

    def plot_agreement_matrix(self, **kwargs):
        from .benchmark_plot_tools import plot_agreement_matrix
        return plot_agreement_matrix(self, **kwargs)

    def plot_performances_vs_snr(self, **kwargs):
        from .benchmark_plot_tools import plot_performances_vs_snr
        return plot_performances_vs_snr(self, **kwargs)


    def plot_error_metrics(self, metric="cosine", case_keys=None, figsize=(15, 5)):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        fig, axes = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):

            result = self.get_result(key)
            scores = result["gt_comparison"].get_ordered_agreement_scores()

            unit_ids1 = scores.index.values
            unit_ids2 = scores.columns.values
            inds_1 = result["gt_comparison"].sorting1.ids_to_indices(unit_ids1)
            inds_2 = result["gt_comparison"].sorting2.ids_to_indices(unit_ids2)
            t1 = result["sliced_gt_templates"].templates_array[:]
            t2 = result["clustering_templates"].templates_array[:]
            a = t1.reshape(len(t1), -1)[inds_1]
            b = t2.reshape(len(t2), -1)[inds_2]

            import sklearn

            if metric == "cosine":
                distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
            else:
                distances = sklearn.metrics.pairwise_distances(a, b, metric)

            im = axes[0, count].imshow(distances, aspect="auto")
            axes[0, count].set_title(metric)
            fig.colorbar(im, ax=axes[0, count])
            label = self.cases[key]["label"]
            axes[0, count].set_title(label)

        return fig

    def plot_metrics_vs_snr(self, metric="agreement", case_keys=None, figsize=(15, 5), axes=None):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        if axes is None:
            fig, axes = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)
            axes = axes.flatten()
        else:
            fig = None

        for count, key in enumerate(case_keys):

            result = self.get_result(key)
            scores = result["gt_comparison"].agreement_scores

            analyzer = self.get_sorting_analyzer(key)
            metrics = analyzer.get_extension("quality_metrics").get_data()

            unit_ids1 = result["gt_comparison"].unit1_ids
            matched_ids2 = result["gt_comparison"].hungarian_match_12.values
            mask = matched_ids2 > -1

            inds_1 = result["gt_comparison"].sorting1.ids_to_indices(unit_ids1[mask])
            inds_2 = result["gt_comparison"].sorting2.ids_to_indices(matched_ids2[mask])

            t1 = result["sliced_gt_templates"].templates_array[:]
            t2 = result["clustering_templates"].templates_array[:]
            a = t1.reshape(len(t1), -1)
            b = t2.reshape(len(t2), -1)

            import sklearn

            if metric == "cosine":
                distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
            elif metric == "l2":
                distances = sklearn.metrics.pairwise_distances(a, b, metric)

            snr_matched = metrics["snr"][unit_ids1[mask]]
            snr_missed = metrics["snr"][unit_ids1[~mask]]

            to_plot = []
            if metric in ["cosine", "l2"]:
                for found, real in zip(inds_2, inds_1):
                    to_plot += [distances[real, found]]
            elif metric == "agreement":
                for found, real in zip(matched_ids2[mask], unit_ids1[mask]):
                    to_plot += [scores.at[real, found]]
            axes[count].plot(snr_matched, to_plot, ".", label="matched")
            axes[count].plot(snr_missed, np.zeros(len(snr_missed)), ".", c="r", label="missed")
            axes[count].set_xlabel("snr")
            axes[count].set_ylabel(metric)
            label = self.cases[key]["label"]
            axes[count].set_title(label)
            axes[count].legend()

        return fig

    def plot_metrics_vs_depth_and_snr(self, metric="agreement", case_keys=None, figsize=(15, 5)):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        fig, axes = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):

            result = self.get_result(key)
            scores = result["gt_comparison"].agreement_scores

            positions = result["sliced_gt_sorting"].get_property("gt_unit_locations")
            # positions = self.datasets[key[1]][1].get_property("gt_unit_locations")
            depth = positions[:, 1]

            analyzer = self.get_sorting_analyzer(key)
            metrics = analyzer.get_extension("quality_metrics").get_data()

            unit_ids1 = result["gt_comparison"].unit1_ids
            matched_ids2 = result["gt_comparison"].hungarian_match_12.values
            mask = matched_ids2 > -1

            inds_1 = result["gt_comparison"].sorting1.ids_to_indices(unit_ids1[mask])
            inds_2 = result["gt_comparison"].sorting2.ids_to_indices(matched_ids2[mask])

            t1 = result["sliced_gt_templates"].templates_array[:]
            t2 = result["clustering_templates"].templates_array[:]
            a = t1.reshape(len(t1), -1)
            b = t2.reshape(len(t2), -1)

            import sklearn

            if metric == "cosine":
                distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
            elif metric == "l2":
                distances = sklearn.metrics.pairwise_distances(a, b, metric)

            snr_matched = metrics["snr"][unit_ids1[mask]]
            snr_missed = metrics["snr"][unit_ids1[~mask]]
            depth_matched = depth[mask]
            depth_missed = depth[~mask]

            to_plot = []
            if metric in ["cosine", "l2"]:
                for found, real in zip(inds_2, inds_1):
                    to_plot += [distances[real, found]]
            elif metric == "agreement":
                for found, real in zip(matched_ids2[mask], unit_ids1[mask]):
                    to_plot += [scores.at[real, found]]
            elif metric in ["recall", "precision", "accuracy"]:
                to_plot = result["gt_comparison"].get_performance()[metric].values
                depth_matched = depth
                snr_matched = metrics["snr"]

            im = axes[0, count].scatter(depth_matched, snr_matched, c=to_plot, label="matched")
            im.set_clim(0, 1)
            axes[0, count].scatter(depth_missed, snr_missed, c=np.zeros(len(snr_missed)), label="missed")
            axes[0, count].set_xlabel("depth")
            axes[0, count].set_ylabel("snr")
            label = self.cases[key]["label"]
            axes[0, count].set_title(label)
            if count > 0:
                axes[0, count].set_ylabel("")
                axes[0, count].set_yticks([], [])
            # axs[0, count].legend()

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.75])
        fig.colorbar(im, cax=cbar_ax, label=metric)

        return fig

    def plot_unit_losses(self, cases_before, cases_after, metric="agreement", figsize=None):

        fig, axs = plt.subplots(ncols=len(cases_before), nrows=1, figsize=figsize)

        for count, (case_before, case_after) in enumerate(zip(cases_before, cases_after)):

            ax = axs[count]
            dataset_key = self.cases[case_before]["dataset"]
            _, gt_sorting1 = self.datasets[dataset_key]
            positions = gt_sorting1.get_property("gt_unit_locations")

            analyzer = self.get_sorting_analyzer(case_before)
            metrics_before = analyzer.get_extension("quality_metrics").get_data()
            x = metrics_before["snr"].values

            y_before = self.get_result(case_before)["gt_comparison"].get_performance()[metric].values
            y_after = self.get_result(case_after)["gt_comparison"].get_performance()[metric].values
            ax.set_ylabel("depth (um)")
            ax.set_ylabel("snr")
            if count > 0:
                ax.set_ylabel("")
                ax.set_yticks([], [])
            im = ax.scatter(positions[:, 1], x, c=(y_after - y_before), cmap="coolwarm")
            im.set_clim(-1, 1)
            # fig.colorbar(im, ax=ax)
            # ax.set_title(k)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.75])
        cbar = fig.colorbar(im, cax=cbar_ax, label=metric)
        # cbar.set_clim(-1, 1)

        return fig

    def plot_comparison_clustering(
        self,
        case_keys=None,
        performance_names=["accuracy", "recall", "precision"],
        colors=["g", "b", "r"],
        ylim=(-0.1, 1.1),
        figsize=None,
    ):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

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

    def plot_some_over_merged(self, case_keys=None, overmerged_score=0.05, max_units=5, figsize=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        figs = []
        for count, key in enumerate(case_keys):
            label = self.cases[key]["label"]
            comp = self.get_result(key)["gt_comparison"]

            unit_index = np.flatnonzero(np.sum(comp.agreement_scores.values > overmerged_score, axis=0) > 1)
            overmerged_ids = comp.sorting2.unit_ids[unit_index]

            n = min(len(overmerged_ids), max_units)
            if n > 0:
                fig, axs = plt.subplots(nrows=n, figsize=figsize)
                for i, unit_id in enumerate(overmerged_ids[:n]):
                    gt_unit_indices = np.flatnonzero(comp.agreement_scores.loc[:, unit_id].values > overmerged_score)
                    gt_unit_ids = comp.sorting1.unit_ids[gt_unit_indices]
                    ax = axs[i]
                    ax.set_title(f"unit {unit_id} - GTids {gt_unit_ids}")

                    analyzer = self.get_sorting_analyzer(key)

                    wf_template = analyzer.get_extension("templates")
                    templates = wf_template.get_templates(unit_ids=gt_unit_ids)
                    if analyzer.sparsity is not None:
                        chan_mask = np.any(analyzer.sparsity.mask[gt_unit_indices, :], axis=0)
                        templates = templates[:, :, chan_mask]
                    ax.plot(templates.swapaxes(1, 2).reshape(templates.shape[0], -1).T)
                    ax.set_xticks([])

                fig.suptitle(label)
                figs.append(fig)
            else:
                print(key, "no overmerged")

        return figs

    def plot_some_over_splited(self, case_keys=None, oversplit_score=0.05, max_units=5, figsize=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        figs = []
        for count, key in enumerate(case_keys):
            label = self.cases[key]["label"]
            comp = self.get_result(key)["gt_comparison"]

            gt_unit_indices = np.flatnonzero(np.sum(comp.agreement_scores.values > oversplit_score, axis=1) > 1)
            oversplit_ids = comp.sorting1.unit_ids[gt_unit_indices]

            n = min(len(oversplit_ids), max_units)
            if n > 0:
                fig, axs = plt.subplots(nrows=n, figsize=figsize)
                for i, unit_id in enumerate(oversplit_ids[:n]):
                    unit_indices = np.flatnonzero(comp.agreement_scores.loc[unit_id, :].values > oversplit_score)
                    unit_ids = comp.sorting2.unit_ids[unit_indices]
                    ax = axs[i]
                    ax.set_title(f"Gt unit {unit_id} - unit_ids: {unit_ids}")

                    templates = self.get_result(key)["clustering_templates"]

                    template_arrays = templates.get_dense_templates()[unit_indices, :, :]
                    if templates.sparsity is not None:
                        chan_mask = np.any(templates.sparsity.mask[gt_unit_indices, :], axis=0)
                        template_arrays = template_arrays[:, :, chan_mask]

                    ax.plot(template_arrays.swapaxes(1, 2).reshape(template_arrays.shape[0], -1).T)
                    ax.set_xticks([])

                fig.suptitle(label)
                figs.append(fig)
            else:
                print(key, "no over splited")

            return figs
