from __future__ import annotations

from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.comparison.comparisontools import make_matching_events
from spikeinterface.core import get_noise_levels

import numpy as np

from .benchmark_base import Benchmark, BenchmarkStudy

class PeakSelectionBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, indices, exhaustive_gt=True):
        pass

    def run(self, **job_kwargs):
        pass

    def compute_result(self, **result_params):
        pass

    # _run_key_saved = [
    #     ("peak_labels", "npy"),
    # ]

    # _result_key_saved = [
    #     ("gt_comparison", "pickle"),
    #     ("sliced_gt_sorting", "sorting"),
    #     ("clustering", "sorting"),
    #     ("sliced_gt_templates", "zarr_templates"),
    #     ("clustering_templates", "zarr_templates"),
    # ]


class PeakSelectionStudy(BenchmarkStudy):

    benchmark_class = PeakSelectionBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        benchmark = PeakSelectionBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark


# class BenchmarkPeakSelection:
#     def __init__(self, recording, gt_sorting, exhaustive_gt=True, job_kwargs={}, tmp_folder=None, verbose=True):
#         self.verbose = verbose
#         self.recording = recording
#         self.gt_sorting = gt_sorting
#         self.job_kwargs = job_kwargs
#         self.exhaustive_gt = exhaustive_gt
#         self.sampling_rate = self.recording.get_sampling_frequency()

#         self.tmp_folder = tmp_folder
#         if self.tmp_folder is None:
#             self.tmp_folder = os.path.join(".", "".join(random.choices(string.ascii_uppercase + string.digits, k=8)))

#         self._peaks = None
#         self._positions = None
#         self._gt_positions = None
#         self.gt_peaks = None

#         self.waveforms = {}
#         self.pcas = {}
#         self.templates = {}

#     def __del__(self):
#         import shutil

#         shutil.rmtree(self.tmp_folder)

#     def set_peaks(self, peaks):
#         self._peaks = peaks

#     def set_positions(self, positions):
#         self._positions = positions

#     @property
#     def peaks(self):
#         if self._peaks is None:
#             self.detect_peaks()
#         return self._peaks

#     @property
#     def positions(self):
#         if self._positions is None:
#             self.localize_peaks()
#         return self._positions

#     @property
#     def gt_positions(self):
#         if self._gt_positions is None:
#             self.localize_gt_peaks()
#         return self._gt_positions

#     def detect_peaks(self, method_kwargs={"method": "locally_exclusive"}):
#         from spikeinterface.sortingcomponents.peak_detection import detect_peaks

#         if self.verbose:
#             method = method_kwargs["method"]
#             print(f"Detecting peaks with method {method}")
#         self._peaks = detect_peaks(self.recording, **method_kwargs, **self.job_kwargs)

#     def localize_peaks(self, method_kwargs={"method": "center_of_mass"}):
#         from spikeinterface.sortingcomponents.peak_localization import localize_peaks

#         if self.verbose:
#             method = method_kwargs["method"]
#             print(f"Localizing peaks with method {method}")
#         self._positions = localize_peaks(self.recording, self.peaks, **method_kwargs, **self.job_kwargs)

#     def localize_gt_peaks(self, method_kwargs={"method": "center_of_mass"}):
#         from spikeinterface.sortingcomponents.peak_localization import localize_peaks

#         if self.verbose:
#             method = method_kwargs["method"]
#             print(f"Localizing gt peaks with method {method}")
#         self._gt_positions = localize_peaks(self.recording, self.gt_peaks, **method_kwargs, **self.job_kwargs)

#     def run(self, peaks=None, positions=None, delta=0.2):
#         t_start = time.time()

#         if peaks is not None:
#             self._peaks = peaks

#         nb_peaks = len(self.peaks)

#         if positions is not None:
#             self._positions = positions

#         spikes1 = self.gt_sorting.to_spike_vector(concatenated=False)[0]["sample_index"]
#         times2 = self.peaks["sample_index"]

#         print("The gt recording has {} peaks and {} have been detected".format(len(times1[0]), len(times2)))

#         matches = make_matching_events(spikes1["sample_index"], times2, int(delta * self.sampling_rate / 1000))
#         self.matches = matches

#         self.deltas = {"labels": [], "delta": matches["delta_frame"]}
#         self.deltas["labels"] = spikes1["unit_index"][matches["index1"]]

#         gt_matches = matches["index1"]
#         self.sliced_gt_sorting = NumpySorting(spikes1[gt_matches], self.sampling_rate, self.gt_sorting.unit_ids)

#         ratio = 100 * len(gt_matches) / len(spikes1)
#         print("Only {0:.2f}% of gt peaks are matched to detected peaks".format(ratio))

#         matches = make_matching_events(times2, spikes1["sample_index"], int(delta * self.sampling_rate / 1000))
#         self.good_matches = matches["index1"]

#         garbage_matches = ~np.isin(np.arange(len(times2)), self.good_matches)
#         garbage_channels = self.peaks["channel_index"][garbage_matches]
#         garbage_peaks = times2[garbage_matches]
#         nb_garbage = len(garbage_peaks)

#         ratio = 100 * len(garbage_peaks) / len(times2)
#         self.garbage_sorting = NumpySorting.from_times_labels(garbage_peaks, garbage_channels, self.sampling_rate)

#         print("The peaks have {0:.2f}% of garbage (without gt around)".format(ratio))

#         self.comp = GroundTruthComparison(self.gt_sorting, self.sliced_gt_sorting, exhaustive_gt=self.exhaustive_gt)

#         for label, sorting in zip(
#             ["gt", "full_gt", "garbage"], [self.sliced_gt_sorting, self.gt_sorting, self.garbage_sorting]
#         ):
#             tmp_folder = os.path.join(self.tmp_folder, label)
#             if os.path.exists(tmp_folder):
#                 import shutil

#                 shutil.rmtree(tmp_folder)

#             if not (label == "full_gt" and label in self.waveforms):
#                 if self.verbose:
#                     print(f"Extracting waveforms for {label}")

#                 self.waveforms[label] = extract_waveforms(
#                     self.recording,
#                     sorting,
#                     tmp_folder,
#                     load_if_exists=True,
#                     ms_before=2.5,
#                     ms_after=3.5,
#                     max_spikes_per_unit=500,
#                     return_scaled=False,
#                     **self.job_kwargs,
#                 )

#                 self.templates[label] = self.waveforms[label].get_all_templates(mode="median")

#         if self.gt_peaks is None:
#             if self.verbose:
#                 print("Computing gt peaks")
#             gt_peaks_ = self.gt_sorting.to_spike_vector()
#             self.gt_peaks = np.zeros(
#                 gt_peaks_.size,
#                 dtype=[
#                     ("sample_index", "<i8"),
#                     ("channel_index", "<i8"),
#                     ("segment_index", "<i8"),
#                     ("amplitude", "<f8"),
#                 ],
#             )
#             self.gt_peaks["sample_index"] = gt_peaks_["sample_index"]
#             self.gt_peaks["segment_index"] = gt_peaks_["segment_index"]
#             max_channels = get_template_extremum_channel(self.waveforms["full_gt"], peak_sign="neg", outputs="index")
#             max_amplitudes = get_template_extremum_amplitude(self.waveforms["full_gt"], peak_sign="neg")

#             for unit_ind, unit_id in enumerate(self.waveforms["full_gt"].sorting.unit_ids):
#                 mask = gt_peaks_["unit_index"] == unit_ind
#                 max_channel = max_channels[unit_id]
#                 self.gt_peaks["channel_index"][mask] = max_channel
#                 self.gt_peaks["amplitude"][mask] = max_amplitudes[unit_id]

#         self.sliced_gt_peaks = self.gt_peaks[gt_matches]
#         self.sliced_gt_positions = self.gt_positions[gt_matches]
#         self.sliced_gt_labels = self.sliced_gt_sorting.to_spike_vector()["unit_index"]
#         self.gt_labels = self.gt_sorting.to_spike_vector()["unit_index"]
#         self.garbage_positions = self.positions[garbage_matches]
#         self.garbage_peaks = self.peaks[garbage_matches]

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
#             else:
#                 ax.scatter(xk, yk, s=s, colorun="k", alpha=alpha, marker=".")

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

#     def plot_clusters(self, title=None, show_probe=False, show_ellipses=True):
#         fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))
#         if title is not None:
#             fig.suptitle(f"Peak selection results with {title}")

#         ax = axs[0]
#         ax.set_title("Full gt clusters")
#         if show_probe:
#             plot_probe_map(self.recording, ax=ax)

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
#             plot_probe_map(self.recording, ax=ax)

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
#         ax.set_title("Garbage")
#         if show_probe:
#             plot_probe_map(self.recording, ax=ax)

#         ax.scatter(self.garbage_positions["x"], self.garbage_positions["y"], c="k", s=1, alpha=0.5)
#         if self.exhaustive_gt:
#             ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
#             ax.set_yticks([], [])
#         ax.set_xlabel("x")

#     def plot_clusters_amplitudes(self, title=None, show_probe=False, clim=(-100, 0), cmap="viridis"):
#         fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 10))
#         if title is not None:
#             fig.suptitle(f"Peak selection results with {title}")

#         ax = axs[0]
#         ax.set_title("Full gt clusters")
#         if show_probe:
#             plot_probe_map(self.recording, ax=ax)

#         from spikeinterface.widgets import get_unit_colors

#         channels = get_template_extremum_channel(self.waveforms["full_gt"], outputs="index")

#         # cb = fig.colorbar(cm, ax=ax)
#         # cb.set_label(metric)

#         import matplotlib

#         my_cmap = plt.colormaps[cmap]
#         cNorm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1])
#         scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

#         for unit_id in self.gt_sorting.unit_ids:
#             wfs = self.waveforms["full_gt"].get_waveforms(unit_id)
#             amplitudes = wfs[:, self.waveforms["full_gt"].nbefore, channels[unit_id]]

#             idx = self.waveforms["full_gt"].get_sampled_indices(unit_id)["spike_index"]
#             all_spikes = self.waveforms["full_gt"].sorting.get_unit_spike_train(unit_id)
#             mask = np.isin(self.gt_peaks["sample_index"], all_spikes[idx])
#             colors = scalarMap.to_rgba(self.gt_peaks["amplitude"][mask])
#             ax.scatter(self.gt_positions["x"][mask], self.gt_positions["y"][mask], c=colors, s=1, alpha=0.5)
#             x_mean, y_mean = (self.gt_positions["x"][mask].mean(), self.gt_positions["y"][mask].mean())
#             ax.annotate(unit_id, (x_mean, y_mean))

#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")

#         ax = axs[1]
#         ax.set_title("Sliced gt clusters")
#         if show_probe:
#             plot_probe_map(self.recording, ax=ax)

#         from spikeinterface.widgets import get_unit_colors

#         channels = get_template_extremum_channel(self.waveforms["gt"], outputs="index")

#         for unit_id in self.sliced_gt_sorting.unit_ids:
#             wfs = self.waveforms["gt"].get_waveforms(unit_id)
#             amplitudes = wfs[:, self.waveforms["gt"].nbefore, channels[unit_id]]

#             idx = self.waveforms["gt"].get_sampled_indices(unit_id)["spike_index"]
#             all_spikes = self.waveforms["gt"].sorting.get_unit_spike_train(unit_id)
#             mask = np.isin(self.sliced_gt_peaks["sample_index"], all_spikes[idx])
#             colors = scalarMap.to_rgba(self.sliced_gt_peaks["amplitude"][mask])
#             ax.scatter(
#                 self.sliced_gt_positions["x"][mask], self.sliced_gt_positions["y"][mask], c=colors, s=1, alpha=0.5
#             )
#             x_mean, y_mean = (self.sliced_gt_positions["x"][mask].mean(), self.sliced_gt_positions["y"][mask].mean())
#             ax.annotate(unit_id, (x_mean, y_mean))

#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.set_xlabel("x")
#         ax.set_yticks([], [])
#         # ax.set_ylabel('y')

#         ax = axs[2]
#         ax.set_title("Garbage")
#         if show_probe:
#             plot_probe_map(self.recording, ax=ax)

#         from spikeinterface.widgets import get_unit_colors

#         channels = get_template_extremum_channel(self.waveforms["garbage"], outputs="index")

#         for unit_id in self.garbage_sorting.unit_ids:
#             wfs = self.waveforms["garbage"].get_waveforms(unit_id)
#             amplitudes = wfs[:, self.waveforms["garbage"].nbefore, channels[unit_id]]

#             idx = self.waveforms["garbage"].get_sampled_indices(unit_id)["spike_index"]
#             all_spikes = self.waveforms["garbage"].sorting.get_unit_spike_train(unit_id)
#             mask = np.isin(self.garbage_peaks["sample_index"], all_spikes[idx])
#             colors = scalarMap.to_rgba(self.garbage_peaks["amplitude"][mask])
#             ax.scatter(self.garbage_positions["x"][mask], self.garbage_positions["y"][mask], c=colors, s=1, alpha=0.5)
#             x_mean, y_mean = (self.garbage_positions["x"][mask].mean(), self.garbage_positions["y"][mask].mean())
#             ax.annotate(unit_id, (x_mean, y_mean))

#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.set_xlabel("x")
#         ax.set_yticks([], [])
#         # ax.set_ylabel('y')

#     def plot_statistics(self, metric="cosine", annotations=True, detect_threshold=5):
#         fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))

#         ax = axs[0, 0]
#         plot_agreement_matrix(self.comp, ax=ax)

#         scores = self.comp.get_ordered_agreement_scores()
#         unit_ids1 = scores.index.values
#         unit_ids2 = scores.columns.values
#         inds_1 = self.comp.sorting1.ids_to_indices(unit_ids1)
#         inds_2 = self.comp.sorting2.ids_to_indices(unit_ids2)

#         a = self.templates["full_gt"].reshape(len(self.templates["full_gt"]), -1)[inds_1]
#         b = self.templates["gt"].reshape(len(self.templates["gt"]), -1)[inds_2]

#         import sklearn

#         if metric == "cosine":
#             distances = sklearn.metrics.pairwise.cosine_similarity(a, b)
#         else:
#             distances = sklearn.metrics.pairwise_distances(a, b, metric)

#         print(distances)
#         ax = axs[0, 1]
#         im = ax.imshow(distances, aspect="auto")
#         ax.set_title(metric)
#         fig.colorbar(im, ax=ax)

#         ax.set_yticks(np.arange(0, len(scores.index)))
#         ax.set_yticklabels(scores.index, fontsize=8)

#         ax.set_xticks(np.arange(0, len(scores.columns)))
#         ax.set_xticklabels(scores.columns, fontsize=8)

#         ax = axs[0, 2]

#         ax.set_ylabel("Time mismatch (time step)")
#         for unit_ind, unit_id in enumerate(self.gt_sorting.unit_ids):
#             mask = self.deltas["labels"] == unit_id
#             ax.violinplot(
#                 self.deltas["delta"][mask], [unit_ind], widths=2, showmeans=True, showmedians=False, showextrema=False
#             )
#         ax.set_xticks(np.arange(len(self.gt_sorting.unit_ids)), self.gt_sorting.unit_ids)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#         ax = axs[1, 0]

#         noise_levels = get_noise_levels(self.recording, return_scaled=False)
#         snrs = self.peaks["amplitude"] / noise_levels[self.peaks["channel_index"]]
#         garbage_snrs = self.garbage_peaks["amplitude"] / noise_levels[self.garbage_peaks["channel_index"]]
#         amin, amax = snrs.min(), snrs.max()

#         ax.hist(snrs, np.linspace(amin, amax, 100), density=True, label="peaks")
#         # ax.hist(garbage_snrs, np.linspace(amin, amax, 100), density=True, label='garbage', alpha=0.5)
#         ax.hist(
#             self.sliced_gt_peaks["amplitude"] / noise_levels[self.sliced_gt_peaks["channel_index"]],
#             np.linspace(amin, amax, 100),
#             density=True,
#             alpha=0.5,
#             label="matched gt",
#         )
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.legend()
#         ax.set_xlabel("snrs")
#         ax.set_ylabel("density")

#         # dist = []
#         # dist_real = []

#         # for found, real in zip(unit_ids2, unit_ids1):
#         #     wfs = self.waveforms['gt'].get_waveforms(found)
#         #     wfs_real = self.waveforms['full_gt'].get_waveforms(real)
#         #     template = self.waveforms['gt'].get_template(found)
#         #     template_real = self.waveforms['full_gt'].get_template(real)

#         #     template = template.reshape(template.size, 1).T
#         #     template_real = template_real.reshape(template_real.size, 1).T

#         #     if metric == 'cosine':
#         #         dist += [sklearn.metrics.pairwise.cosine_similarity(template, wfs.reshape(len(wfs), -1), metric).flatten()]
#         #         dist_real += [sklearn.metrics.pairwise.cosine_similarity(template_real, wfs_real.reshape(len(wfs_real), -1), metric).flatten()]
#         #     else:
#         #         dist += [sklearn.metrics.pairwise_distances(template, wfs.reshape(len(wfs), -1), metric).flatten()]
#         #         dist_real += [sklearn.metrics.pairwise_distances(template_real, wfs_real.reshape(len(wfs_real), -1), metric).flatten()]

#         # ax.errorbar([a.mean() for a in dist], [a.mean() for a in dist_real], [a.std() for a in dist], [a.std() for a in dist_real], capsize=0, ls='none', color='black',
#         #     elinewidth=2)
#         # ax.plot([0, 1], [0, 1], '--')
#         # ax.set_xlabel('cosine dispersion tested')
#         # ax.set_ylabel('cosine dispersion gt')
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)

#         ax = axs[1, 1]
#         nb_spikes_real = []
#         nb_spikes = []

#         for found, real in zip(unit_ids2, unit_ids1):
#             a = self.gt_sorting.get_unit_spike_train(real).size
#             b = self.sliced_gt_sorting.get_unit_spike_train(found).size
#             nb_spikes_real += [a]
#             nb_spikes += [b]

#         centers = compute_center_of_mass(self.waveforms["gt"])
#         spikes_seg0 = self.sliced_gt_sorting.to_spike_vector(concatenated=False)[0]
#         stds = []
#         means = []
#         for found, real in zip(inds_2, inds_1):
#             mask = spikes_seg0["unit_index"] == found
#             center = np.array([self.sliced_gt_positions[mask]["x"], self.sliced_gt_positions[mask]["y"]]).mean()
#             means += [np.mean(center - centers[real])]
#             stds += [np.std(center - centers[real])]

#         metrics = compute_quality_metrics(self.waveforms["full_gt"], metric_names=["snr"], load_if_exists=False)
#         ax.errorbar(metrics["snr"][inds_1], means, yerr=stds, ls="none")

#         ax.set_xlabel("template snr")
#         ax.set_ylabel("position error")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#         if annotations:
#             for l, x, y in zip(unit_ids1, metrics["snr"][inds_1], means):
#                 ax.annotate(l, (x, y))

#         if detect_threshold is not None:
#             ymin, ymax = ax.get_ylim()
#             ax.plot([detect_threshold, detect_threshold], [ymin, ymax], "k--")

#         # ax.plot(nb_spikes, nb_spikes_real, '.', markersize=10)
#         # ax.set_xlabel("# spikes tested")
#         # ax.set_ylabel("# spikes gt")
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
#         # xmin, xmax = ax.get_xlim()
#         # ymin, ymax = ax.get_ylim()
#         # ax.plot([xmin, xmax], [xmin, xmin + (xmax - xmin)], 'k--')

#         # if annotations:
#         #     for l,x,y in zip(unit_ids1, nb_spikes, nb_spikes_real):
#         #         ax.annotate(l, (x, y))

#         # fs = self.recording_f.get_sampling_frequency()
#         # tmax = self.recording_f.get_total_duration()
#         # ax.hist(self.peaks['sample_index']/fs, np.linspace(0, tmax, 100), density=True)
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
#         # ax.set_xlabel('time (s)')
#         # ax.set_ylabel('density')

#         # for channel_index in
#         #     ax.hist(snrs, np.linspace(amin, amax, 100), density=True, label='peaks')
#         # ax.spines['top'].set_visible(False)
#         # ax.spines['right'].set_visible(False)
#         # ax.legend()
#         # ax.set_xlabel('snrs')
#         # ax.set_ylabel('density')

#         ax = axs[1, 2]
#         metrics = compute_quality_metrics(self.waveforms["full_gt"], metric_names=["snr"], load_if_exists=False)
#         ratios = np.array(nb_spikes) / np.array(nb_spikes_real)
#         plt.plot(metrics["snr"][inds_1], ratios, ".")
#         ax.set_xlabel("template snr")
#         ax.set_ylabel("% gt spikes detected")
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#         if annotations:
#             for l, x, y in zip(unit_ids1, metrics["snr"][inds_1], ratios):
#                 ax.annotate(l, (x, y))

#         if detect_threshold is not None:
#             ymin, ymax = ax.get_ylim()
#             ax.plot([detect_threshold, detect_threshold], [ymin, ymax], "k--")

#     def explore_garbage(self, channel_index, nb_bins=None, dt=None):
#         mask = self.garbage_peaks["channel_index"] == channel_index
#         times2 = self.garbage_peaks[mask]["sample_index"]
#         spikes1 = self.gt_sorting.to_spike_vector(concatenate=False)[0]

#         from spikeinterface.comparison.comparisontools import make_matching_events

#         if dt is None:
#             delta = self.waveforms["garbage"].nafter
#         else:
#             delta = dt
#         matches = make_matching_events(times2, spikes1["sample_index"], delta)
#         unit_inds = spikes1["unit_index"][matches["index2"]]
#         dt = matches["delta_frame"]
#         res = {}
#         fig, ax = plt.subplots()
#         if nb_bins is None:
#             nb_bins = 2 * delta
#         xaxis = np.linspace(-delta, delta, nb_bins)
#         for unit_ind in np.unique(unit_inds):
#             mask = unit_inds == unit_ind
#             res[unit_ind] = dt[mask]
#             ax.hist(res[unit_ind], bins=xaxis)
