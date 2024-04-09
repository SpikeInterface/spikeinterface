from __future__ import annotations

from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import (
    plot_probe_map,
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
    plot_unit_templates,
    plot_unit_waveforms,
)
from spikeinterface.comparison.comparisontools import make_matching_events

import matplotlib.patches as mpatches

# from spikeinterface.postprocessing import get_template_extremum_channel
from spikeinterface.core import get_noise_levels

import pylab as plt
import numpy as np


from .benchmark_tools import BenchmarkStudy, Benchmark
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.template_tools import get_template_extremum_channel


class ClusteringBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, indices, exhaustive_gt=True):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.indices = indices

        sorting_analyzer = create_sorting_analyzer(self.gt_sorting, self.recording, format="memory", sparse=False)
        sorting_analyzer.compute(["random_spikes", "templates"])
        extremum_channel_inds = get_template_extremum_channel(sorting_analyzer, outputs="index")

        peaks = self.gt_sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        if self.indices is None:
            self.indices = np.arange(len(peaks))
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
        data["unit_index"] = self.result["peak_labels"][~self.noise]

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

    def plot_agreements(self, case_keys=None, figsize=(15, 15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            ax.set_title(self.cases[key]["label"])
            plot_agreement_matrix(self.get_result(key)["gt_comparison"], ax=ax)

    def plot_performances_vs_snr(self, case_keys=None, figsize=(15, 15)):
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

    def plot_error_metrics(self, metric="cosine", case_keys=None, figsize=(15, 5)):

        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):

            result = self.get_result(key)
            scores = result["gt_comparison"].get_ordered_agreement_scores()

            unit_ids1 = scores.index.values
            unit_ids2 = scores.columns.values
            inds_1 = result["gt_comparison"].sorting1.ids_to_indices(unit_ids1)
            inds_2 = result["gt_comparison"].sorting2.ids_to_indices(unit_ids2)
            t1 = result["sliced_gt_templates"].templates_array
            t2 = result["clustering_templates"].templates_array
            a = t1.reshape(len(t1), -1)[inds_1]
            b = t2.reshape(len(t2), -1)[inds_2]

            import sklearn

            if metric == "cosine":
                distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
            else:
                distances = sklearn.metrics.pairwise_distances(a, b, metric)

            im = axs[count].imshow(distances, aspect="auto")
            axs[count].set_title(metric)
            fig.colorbar(im, ax=axs[count])
            label = self.cases[key]["label"]
            axs[0, count].set_title(label)

    def plot_metrics_vs_snr(self, metric="cosine", case_keys=None, figsize=(15, 5)):

        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):

            result = self.get_result(key)
            scores = result["gt_comparison"].get_ordered_agreement_scores()

            analyzer = self.get_sorting_analyzer(key)
            metrics = analyzer.get_extension("quality_metrics").get_data()

            unit_ids1 = scores.index.values
            unit_ids2 = scores.columns.values
            inds_1 = result["gt_comparison"].sorting1.ids_to_indices(unit_ids1)
            inds_2 = result["gt_comparison"].sorting2.ids_to_indices(unit_ids2)
            t1 = result["sliced_gt_templates"].templates_array
            t2 = result["clustering_templates"].templates_array
            a = t1.reshape(len(t1), -1)
            b = t2.reshape(len(t2), -1)

            import sklearn

            if metric == "cosine":
                distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
            else:
                distances = sklearn.metrics.pairwise_distances(a, b, metric)

            snr = metrics["snr"][unit_ids1][inds_1[: len(inds_2)]]
            to_plot = []
            for found, real in zip(inds_2, inds_1):
                to_plot += [distances[real, found]]
            axs[0, count].plot(snr, to_plot, ".")
            axs[0, count].set_xlabel("snr")
            axs[0, count].set_ylabel(metric)
            label = self.cases[key]["label"]
            axs[0, count].set_title(label)

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


#     def _scatter_clusters(
#         self,
#         xs,
#         ys,
#         sorting,
#         colors=None,
#         labels=None,
#         ax=None,
#         n_std=2.0,
#         force_black_for=[],
#         s=1,
#         alpha=0.5,
#         show_ellipses=True,
#     ):
#         if colors is None:
#             from spikeinterface.widgets import get_unit_colors

#             colors = get_unit_colors(sorting)

#         from matplotlib.patches import Ellipse
#         import matplotlib.transforms as transforms

#         ax = ax or plt.gca()
#         # scatter and collect gaussian info
#         means = {}
#         covs = {}
#         labels = sorting.to_spike_vector(concatenated=False)[0]["unit_index"]

#         for unit_ind, unit_id in enumerate(sorting.unit_ids):
#             where = np.flatnonzero(labels == unit_ind)

#             xk = xs[where]
#             yk = ys[where]

#             if unit_id not in force_black_for:
#                 ax.scatter(xk, yk, s=s, color=colors[unit_id], alpha=alpha, marker=".")
#                 x_mean, y_mean = xk.mean(), yk.mean()
#                 xycov = np.cov(xk, yk)
#                 means[unit_id] = x_mean, y_mean
#                 covs[unit_id] = xycov
#                 ax.annotate(unit_id, (x_mean, y_mean))
#                 ax.scatter([x_mean], [y_mean], s=50, c="k")
#             else:
#                 ax.scatter(xk, yk, s=s, color="k", alpha=alpha, marker=".")

#         for unit_id in means.keys():
#             mean_x, mean_y = means[unit_id]
#             cov = covs[unit_id]

#             with np.errstate(invalid="ignore"):
#                 vx, vy = cov[0, 0], cov[1, 1]
#                 rho = cov[0, 1] / np.sqrt(vx * vy)
#             if not np.isfinite([vx, vy, rho]).all():
#                 continue

#             if show_ellipses:
#                 ell = Ellipse(
#                     (0, 0),
#                     width=2 * np.sqrt(1 + rho),
#                     height=2 * np.sqrt(1 - rho),
#                     facecolor=(0, 0, 0, 0),
#                     edgecolor=colors[unit_id],
#                     linewidth=1,
#                 )
#                 transform = (
#                     transforms.Affine2D()
#                     .rotate_deg(45)
#                     .scale(n_std * np.sqrt(vx), n_std * np.sqrt(vy))
#                     .translate(mean_x, mean_y)
#                 )
#                 ell.set_transform(transform + ax.transData)
#                 ax.add_patch(ell)

#     def plot_clusters(self, show_probe=True, show_ellipses=True):
#         fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))
#         fig.suptitle(f"Clustering results with {self.method}")
#         ax = axs[0]
#         ax.set_title("Full gt clusters")
#         if show_probe:
#             plot_probe_map(self.recording_f, ax=ax)

#         from spikeinterface.widgets import get_unit_colors

#         colors = get_unit_colors(self.gt_sorting)
#         self._scatter_clusters(
#             self.gt_positions["x"],
#             self.gt_positions["y"],
#             self.gt_sorting,
#             colors,
#             s=1,
#             alpha=0.5,
#             ax=ax,
#             show_ellipses=show_ellipses,
#         )
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")

#         ax = axs[1]
#         ax.set_title("Sliced gt clusters")
#         if show_probe:
#             plot_probe_map(self.recording_f, ax=ax)

#         self._scatter_clusters(
#             self.sliced_gt_positions["x"],
#             self.sliced_gt_positions["y"],
#             self.sliced_gt_sorting,
#             colors,
#             s=1,
#             alpha=0.5,
#             ax=ax,
#             show_ellipses=show_ellipses,
#         )
#         if self.exhaustive_gt:
#             ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
#         ax.set_xlabel("x")
#         ax.set_yticks([], [])

#         ax = axs[2]
#         ax.set_title("Found clusters")
#         if show_probe:
#             plot_probe_map(self.recording_f, ax=ax)
#         ax.scatter(self.positions["x"][self.noise], self.positions["y"][self.noise], c="k", s=1, alpha=0.1)
#         self._scatter_clusters(
#             self.positions["x"][~self.noise],
#             self.positions["y"][~self.noise],
#             self.clustering,
#             s=1,
#             alpha=0.5,
#             ax=ax,
#             show_ellipses=show_ellipses,
#         )

#         ax.set_xlabel("x")
#         if self.exhaustive_gt:
#             ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
#             ax.set_yticks([], [])

#     def plot_found_clusters(self, show_probe=True, show_ellipses=True):
#         fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
#         fig.suptitle(f"Clustering results with {self.method}")
#         ax.set_title("Found clusters")
#         if show_probe:
#             plot_probe_map(self.recording_f, ax=ax)
#         ax.scatter(self.positions["x"][self.noise], self.positions["y"][self.noise], c="k", s=1, alpha=0.1)
#         self._scatter_clusters(
#             self.positions["x"][~self.noise],
#             self.positions["y"][~self.noise],
#             self.clustering,
#             s=1,
#             alpha=0.5,
#             ax=ax,
#             show_ellipses=show_ellipses,
#         )

#         ax.set_xlabel("x")
#         if self.exhaustive_gt:
#             ax.set_yticks([], [])

#     def plot_statistics(self, metric="cosine", annotations=True, detect_threshold=5):
#         fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))

#         fig.suptitle(f"Clustering results with {self.method}")
#         metrics = compute_quality_metrics(self.waveforms["gt"], metric_names=["snr"], load_if_exists=False)

#         ax = axs[0, 0]
#         plot_agreement_matrix(self.comp, ax=ax)
#         scores = self.comp.get_ordered_agreement_scores()
#         ymin, ymax = ax.get_ylim()
#         xmin, xmax = ax.get_xlim()
#         unit_ids1 = scores.index.values
#         unit_ids2 = scores.columns.values
#         inds_1 = self.comp.sorting1.ids_to_indices(unit_ids1)
#         snrs = metrics["snr"][inds_1]

#         nb_detectable = len(unit_ids1)

#         if detect_threshold is not None:
#             for count, snr in enumerate(snrs):
#                 if snr < detect_threshold:
#                     ax.plot([xmin, xmax], [count, count], "k")
#                     nb_detectable -= 1

#         ax.plot([nb_detectable + 0.5, nb_detectable + 0.5], [ymin, ymax], "r")

#         # import MEArec as mr
#         # mearec_recording = mr.load_recordings(self.mearec_file)
#         # positions = mearec_recording.template_locations[:]

#         # self.found_positions = np.zeros((len(self.labels), 2))
#         # for i in range(len(self.labels)):
#         #     data = self.positions[self.selected_peaks_labels == self.labels[i]]
#         #     self.found_positions[i] = np.median(data['x']), np.median(data['y'])

#         unit_ids1 = scores.index.values
#         unit_ids2 = scores.columns.values
#         inds_1 = self.comp.sorting1.ids_to_indices(unit_ids1)
#         inds_2 = self.comp.sorting2.ids_to_indices(unit_ids2)

#         a = self.templates["gt"].reshape(len(self.templates["gt"]), -1)[inds_1]
#         b = self.templates["clustering"].reshape(len(self.templates["clustering"]), -1)[inds_2]

#         import sklearn

#         if metric == "cosine":
#             distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
#         else:
#             distances = sklearn.metrics.pairwise_distances(a, b, metric)

#         ax = axs[0, 1]
#         nb_peaks = np.array(
#             [len(self.sliced_gt_sorting.get_unit_spike_train(i)) for i in self.sliced_gt_sorting.unit_ids]
#         )

#         nb_potentials = np.sum(scores.max(1).values > 0.1)

#         ax.plot(
#             metrics["snr"][unit_ids1][inds_1[:nb_potentials]],
#             nb_peaks[inds_1[:nb_potentials]],
#             markersize=10,
#             marker=".",
#             ls="",
#             c="k",
#             label="Cluster potentially found",
#         )
#         ax.plot(
#             metrics["snr"][unit_ids1][inds_1[nb_potentials:]],
#             nb_peaks[inds_1[nb_potentials:]],
#             markersize=10,
#             marker=".",
#             ls="",
#             c="r",
#             label="Cluster clearly missed",
#         )

#         if annotations:
#             for l, x, y in zip(
#                 unit_ids1[: len(inds_2)],
#                 metrics["snr"][unit_ids1][inds_1[: len(inds_2)]],
#                 nb_peaks[inds_1[: len(inds_2)]],
#             ):
#                 ax.annotate(l, (x, y))

#             for l, x, y in zip(
#                 unit_ids1[len(inds_2) :],
#                 metrics["snr"][unit_ids1][inds_1[len(inds_2) :]],
#                 nb_peaks[inds_1[len(inds_2) :]],
#             ):
#                 ax.annotate(l, (x, y), c="r")

#         if detect_threshold is not None:
#             ymin, ymax = ax.get_ylim()
#             ax.plot([detect_threshold, detect_threshold], [ymin, ymax], "k--")

#         ax.legend()
#         ax.set_xlabel("template snr")
#         ax.set_ylabel("nb spikes")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#         ax = axs[0, 2]
#         im = ax.imshow(distances, aspect="auto")
#         ax.set_title(metric)
#         fig.colorbar(im, ax=ax)

#         if detect_threshold is not None:
#             for count, snr in enumerate(snrs):
#                 if snr < detect_threshold:
#                     ax.plot([xmin, xmax], [count, count], "w")

#             ymin, ymax = ax.get_ylim()
#             ax.plot([nb_detectable + 0.5, nb_detectable + 0.5], [ymin, ymax], "r")

#         ax.set_yticks(np.arange(0, len(scores.index)))
#         ax.set_yticklabels(scores.index, fontsize=8)

#         res = []
#         nb_spikes = []
#         energy = []
#         nb_channels = []

#         noise_levels = get_noise_levels(self.recording_f, return_scaled=False)

#         for found, real in zip(unit_ids2, unit_ids1):
#             wfs = self.waveforms["clustering"].get_waveforms(found)
#             wfs_real = self.waveforms["gt"].get_waveforms(real)
#             template = self.waveforms["clustering"].get_template(found)
#             template_real = self.waveforms["gt"].get_template(real)
#             nb_channels += [np.sum(np.std(template_real, 0) < noise_levels)]

#             wfs = wfs.reshape(len(wfs), -1)
#             template = template.reshape(template.size, 1).T
#             template_real = template_real.reshape(template_real.size, 1).T

#             if metric == "cosine":
#                 dist = sklearn.metrics.pairwise.cosine_similarity(template, template_real).flatten().tolist()
#             else:
#                 dist = sklearn.metrics.pairwise_distances(template, template_real, metric).flatten().tolist()
#             res += dist
#             nb_spikes += [self.sliced_gt_sorting.get_unit_spike_train(real).size]
#             energy += [np.linalg.norm(template_real)]

#         ax = axs[1, 0]
#         res = np.array(res)
#         nb_spikes = np.array(nb_spikes)
#         nb_channels = np.array(nb_channels)
#         energy = np.array(energy)

#         snrs = metrics["snr"][unit_ids1][inds_1[: len(inds_2)]]
#         cm = ax.scatter(snrs, nb_spikes, c=res)
#         ax.set_xlabel("template snr")
#         ax.set_ylabel("nb spikes")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         cb = fig.colorbar(cm, ax=ax)
#         cb.set_label(metric)
#         if detect_threshold is not None:
#             ymin, ymax = ax.get_ylim()
#             ax.plot([detect_threshold, detect_threshold], [ymin, ymax], "k--")

#         if annotations:
#             for l, x, y in zip(unit_ids1[: len(inds_2)], snrs, nb_spikes):
#                 ax.annotate(l, (x, y))

#         ax = axs[1, 1]
#         cm = ax.scatter(energy, nb_channels, c=res)
#         ax.set_xlabel("template energy")
#         ax.set_ylabel("nb channels")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         cb = fig.colorbar(cm, ax=ax)
#         cb.set_label(metric)

#         if annotations:
#             for l, x, y in zip(unit_ids1[: len(inds_2)], energy, nb_channels):
#                 ax.annotate(l, (x, y))

#         ax = axs[1, 2]
#         for performance_name in ["accuracy", "recall", "precision"]:
#             perf = self.comp.get_performance()[performance_name]
#             ax.plot(metrics["snr"], perf, markersize=10, marker=".", ls="", label=performance_name)
#         ax.set_xlabel("template snr")
#         ax.set_ylabel("performance")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.legend()
#         if detect_threshold is not None:
#             ymin, ymax = ax.get_ylim()
#             ax.plot([detect_threshold, detect_threshold], [ymin, ymax], "k--")

#         plt.tight_layout()
