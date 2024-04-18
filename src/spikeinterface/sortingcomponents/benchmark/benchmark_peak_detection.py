from __future__ import annotations

from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import (
    plot_probe_map,
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
    plot_unit_templates,
    plot_unit_waveforms,
)
from spikeinterface.comparison.comparisontools import make_matching_events
from spikeinterface.core import get_noise_levels

import time
import string, random
import pylab as plt
import os
import numpy as np

from .benchmark_tools import BenchmarkStudy, Benchmark
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.template_tools import get_template_extremum_channel


class PeakDetectionBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, exhaustive_gt=True):
        self.recording = recording
        self.gt_sorting = gt_sorting

        sorting_analyzer = create_sorting_analyzer(self.gt_sorting, self.recording, format="memory", sparse=False)
        sorting_analyzer.compute(["random_spikes", "templates", "spike_amplitudes"])
        extremum_channel_inds = get_template_extremum_channel(sorting_analyzer, outputs="index")
        self.gt_peaks = self.gt_sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)
        self.params = params
        self.exhaustive_gt = exhaustive_gt
        self.method = params["method"]
        self.method_kwargs = params["method_kwargs"]
        self.result = {"gt_peaks": self.gt_peaks}
        self.result["gt_amplitudes"] = sorting_analyzer.get_extension("spike_amplitudes").get_data()

    def run(self, **job_kwargs):
        peaks = detect_peaks(self.recording, method=self.method, **self.method_kwargs, **job_kwargs)
        self.result["peaks"] = peaks

    def compute_result(self, **result_params):
        spikes = self.result["peaks"]
        self.result["peak_on_channels"] = NumpySorting.from_peaks(
            spikes, self.recording.sampling_frequency, unit_ids=self.recording.channel_ids
        )
        spikes = self.result["gt_peaks"]
        self.result["gt_on_channels"] = NumpySorting.from_peaks(
            spikes, self.recording.sampling_frequency, unit_ids=self.recording.channel_ids
        )

        self.result["gt_comparison"] = GroundTruthComparison(
            self.result["gt_on_channels"], self.result["peak_on_channels"], exhaustive_gt=self.exhaustive_gt
        )

        gt_peaks = self.gt_sorting.to_spike_vector()
        times1 = self.result["gt_peaks"]["sample_index"]
        times2 = self.result["peaks"]["sample_index"]

        print("The gt recording has {} peaks and {} have been detected".format(len(times1), len(times2)))

        matches = make_matching_events(times1, times2, int(0.4 * self.recording.sampling_frequency / 1000))
        self.matches = matches
        self.gt_matches = matches["index1"]

        self.deltas = {"labels": [], "channels": [], "delta": matches["delta_frame"]}
        self.deltas["labels"] = gt_peaks["unit_index"][self.gt_matches]
        self.deltas["channels"] = self.result["gt_peaks"]["unit_index"][self.gt_matches]

        self.result["sliced_gt_sorting"] = NumpySorting(
            gt_peaks[self.gt_matches], self.recording.sampling_frequency, self.gt_sorting.unit_ids
        )

        ratio = 100 * len(self.gt_matches) / len(times1)
        print("Only {0:.2f}% of gt peaks are matched to detected peaks".format(ratio))

        # matches = make_matching_events(times2, times1, int(delta * self.sampling_rate / 1000))
        # self.good_matches = matches["index1"]

        # garbage_matches = ~np.isin(np.arange(len(times2)), self.good_matches)
        # garbage_channels = self.peaks["channel_index"][garbage_matches]
        # garbage_peaks = times2[garbage_matches]
        # nb_garbage = len(garbage_peaks)

        # ratio = 100 * len(garbage_peaks) / len(times2)
        # self.garbage_sorting = NumpySorting.from_times_labels(garbage_peaks, garbage_channels, self.sampling_rate)

        # print("The peaks have {0:.2f}% of garbage (without gt around)".format(ratio))

    _run_key_saved = [("peaks", "npy"), ("gt_peaks", "npy"), ("gt_amplitudes", "npy")]

    _result_key_saved = [
        ("gt_comparison", "pickle"),
        ("sliced_gt_sorting", "sorting"),
        ("peak_on_channels", "sorting"),
        ("gt_on_channels", "sorting"),
    ]


class PeakDetectionStudy(BenchmarkStudy):

    benchmark_class = PeakDetectionBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        benchmark = PeakDetectionBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark

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

    def plot_detected_amplitudes(self, case_keys=None, figsize=(15, 5)):

        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            data1 = self.get_result(key)["peaks"]["amplitude"]
            data2 = self.get_result(key)["gt_amplitudes"]
            bins = np.linspace(data2.min(), data2.max(), 100)
            ax.hist(data1, bins=bins, alpha=0.5, label="detected")
            ax.hist(data2, bins=bins, alpha=0.5, label="gt")
            ax.set_title(self.cases[key]["label"])
            ax.legend()


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
