from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
import shutil

from spikeinterface.core import extract_waveforms, precompute_sparsity, WaveformExtractor


from spikeinterface.extractors import read_mearec
from spikeinterface.preprocessing import bandpass_filter, zscore, common_reference, scale, highpass_filter, whiten
from spikeinterface.sorters import run_sorter, read_sorter_folder

from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.sortingcomponents.motion_interpolation import InterpolateMotionRecording
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import BenchmarkBase, _simpleaxis
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.widgets import plot_sorting_performance
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.curation import MergeUnitsSorting
from spikeinterface.core import get_template_extremum_channel

import sklearn

import matplotlib.pyplot as plt

import MEArec as mr


class BenchmarkMotionInterpolationMearec(BenchmarkBase):
    _array_names = ("gt_motion", "estimated_motion", "temporal_bins", "spatial_bins")
    _waveform_names = ("static", "drifting", "corrected_gt", "corrected_estimated")
    _sorting_names = ()

    _array_names_from_parent = ()
    _waveform_names_from_parent = ("static", "drifting")
    _sorting_names_from_parent = ("static", "drifting")

    def __init__(
        self,
        mearec_filename_drifting,
        mearec_filename_static,
        gt_motion,
        estimated_motion,
        temporal_bins,
        spatial_bins,
        do_preprocessing=True,
        correct_motion_kwargs={},
        waveforms_kwargs=dict(
            ms_before=1.0,
            ms_after=3.0,
            max_spikes_per_unit=500,
        ),
        sparse_kwargs=dict(
            method="radius",
            radius_um=100.0,
        ),
        sorter_cases={},
        folder=None,
        title="",
        job_kwargs={"chunk_duration": "1s", "n_jobs": -1, "progress_bar": True, "verbose": True},
        overwrite=False,
        delete_output_folder=True,
        parent_benchmark=None,
    ):
        BenchmarkBase.__init__(
            self,
            folder=folder,
            title=title,
            overwrite=overwrite,
            job_kwargs=job_kwargs,
            parent_benchmark=parent_benchmark,
        )

        self._args.extend([str(mearec_filename_drifting), str(mearec_filename_static), None, None, None, None])

        self.sorter_cases = sorter_cases.copy()
        self.mearec_filenames = {}
        self.keys = ["static", "drifting", "corrected_gt", "corrected_estimated"]
        self.mearec_filenames["drifting"] = mearec_filename_drifting
        self.mearec_filenames["static"] = mearec_filename_static

        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins
        self.gt_motion = gt_motion
        self.estimated_motion = estimated_motion
        self.do_preprocessing = do_preprocessing
        self.delete_output_folder = delete_output_folder

        self._recordings = None
        _, self.sorting_gt = read_mearec(self.mearec_filenames["static"])

        self.correct_motion_kwargs = correct_motion_kwargs.copy()
        self.sparse_kwargs = sparse_kwargs.copy()
        self.waveforms_kwargs = waveforms_kwargs.copy()
        self.comparisons = {}
        self.accuracies = {}

        self._kwargs.update(
            dict(
                correct_motion_kwargs=self.correct_motion_kwargs,
                sorter_cases=self.sorter_cases,
                do_preprocessing=do_preprocessing,
                delete_output_folder=delete_output_folder,
                waveforms_kwargs=waveforms_kwargs,
                sparse_kwargs=sparse_kwargs,
            )
        )

    @property
    def recordings(self):
        if self._recordings is None:
            self._recordings = {}

            for key in (
                "drifting",
                "static",
            ):
                rec, _ = read_mearec(self.mearec_filenames[key])
                self._recordings["raw_" + key] = rec

                if self.do_preprocessing:
                    # this processing chain is the same as the kilosort2.5
                    # this is important if we want to skip the kilosort preprocessing
                    #   * all computation are done in float32
                    #   * 150um is more or less 30 channels for the whittening
                    #   * the lastet gain step is super important it is what KS2.5 is doing because the whiten traces
                    #     have magnitude around 1 so a factor (200) is needed to go back to int16
                    rec = common_reference(rec, dtype="float32")
                    rec = highpass_filter(rec, freq_min=150.0)
                    rec = whiten(rec, mode="local", radius_um=150.0, num_chunks_per_segment=40, chunk_size=32000)
                    rec = scale(rec, gain=200, dtype="int16")
                self._recordings[key] = rec

            rec = self._recordings["drifting"]
            self._recordings["corrected_gt"] = InterpolateMotionRecording(
                rec, self.gt_motion, self.temporal_bins, self.spatial_bins, **self.correct_motion_kwargs
            )

            self._recordings["corrected_estimated"] = InterpolateMotionRecording(
                rec, self.estimated_motion, self.temporal_bins, self.spatial_bins, **self.correct_motion_kwargs
            )

        return self._recordings

    def run(self):
        self.extract_waveforms()
        self.save_to_folder()
        self.run_sorters()
        self.save_to_folder()

    def extract_waveforms(self):
        # the sparsity is estimated on the static recording and propagated to all of then
        if self.parent_benchmark is None:
            wf_kwargs = self.waveforms_kwargs.copy()
            wf_kwargs.pop("max_spikes_per_unit", None)
            sparsity = precompute_sparsity(
                self.recordings["static"],
                self.sorting_gt,
                num_spikes_for_sparsity=200.0,
                unit_batch_size=10000,
                **wf_kwargs,
                **self.sparse_kwargs,
                **self.job_kwargs,
            )
        else:
            sparsity = self.waveforms["static"].sparsity

        for key in self.keys:
            if self.parent_benchmark is not None and key in self._waveform_names_from_parent:
                continue

            waveforms_folder = self.folder / "waveforms" / key
            we = WaveformExtractor.create(
                self.recordings[key],
                self.sorting_gt,
                waveforms_folder,
                mode="folder",
                sparsity=sparsity,
                remove_if_exists=True,
            )
            we.set_params(**self.waveforms_kwargs, return_scaled=True)
            we.run_extract_waveforms(seed=22051977, **self.job_kwargs)
            self.waveforms[key] = we

    def run_sorters(self, skip_already_done=True):
        for case in self.sorter_cases:
            label = case["label"]
            print("run sorter", label)
            sorter_name = case["sorter_name"]
            sorter_params = case["sorter_params"]
            recording = self.recordings[case["recording"]]
            output_folder = self.folder / f"tmp_sortings_{label}"
            if output_folder.exists() and skip_already_done:
                print("already done")
                sorting = read_sorter_folder(output_folder)
            else:
                sorting = run_sorter(
                    sorter_name,
                    recording,
                    output_folder,
                    **sorter_params,
                    delete_output_folder=self.delete_output_folder,
                )
            self.sortings[label] = sorting

    def compute_distances_to_static(self, force=False):
        if hasattr(self, "distances") and not force:
            return self.distances

        self.distances = {}

        n = len(self.waveforms["static"].unit_ids)

        sparsity = self.waveforms["static"].sparsity

        ref_templates = self.waveforms["static"].get_all_templates()

        for key in self.keys:
            if self.parent_benchmark is not None and key in ("drifting", "static"):
                continue

            print(key)
            dist = self.distances[key] = {
                "std": np.zeros(n),
                "norm_std": np.zeros(n),
                "template_norm_distance": np.zeros(n),
                "template_cosine": np.zeros(n),
            }

            templates = self.waveforms[key].get_all_templates()

            extremum_channel = get_template_extremum_channel(self.waveforms["static"], outputs="index")

            for unit_ind, unit_id in enumerate(self.waveforms[key].sorting.unit_ids):
                mask = sparsity.mask[unit_ind, :]
                ref_template = ref_templates[unit_ind][:, mask]
                template = templates[unit_ind][:, mask]

                max_chan = extremum_channel[unit_id]
                max_chan

                max_chan_sparse = list(np.nonzero(mask)[0]).index(max_chan)

                # this is already sparse
                wfs = self.waveforms[key].get_waveforms(unit_id)
                ref_wfs = self.waveforms["static"].get_waveforms(unit_id)

                rms = np.sqrt(np.mean(template**2))
                ref_rms = np.sqrt(np.mean(ref_template**2))
                if rms == 0:
                    print(key, unit_id, unit_ind, rms, ref_rms)

                dist["std"][unit_ind] = np.mean(np.std(wfs, axis=0), axis=(0, 1))
                dist["norm_std"][unit_ind] = np.mean(np.std(wfs, axis=0), axis=(0, 1)) / rms
                dist["template_norm_distance"][unit_ind] = np.sum((ref_template - template) ** 2) / ref_rms
                dist["template_cosine"][unit_ind] = sklearn.metrics.pairwise.cosine_similarity(
                    ref_template.reshape(1, -1), template.reshape(1, -1)
                )[0]

        return self.distances

    def compute_residuals(self, force=True):
        fr = int(self.recordings["static"].get_sampling_frequency())
        duration = int(self.recordings["static"].get_total_duration())

        t_start = 0
        t_stop = duration

        if hasattr(self, "residuals") and not force:
            return self.residuals, (t_start, t_stop)

        self.residuals = {}

        for key in ["corrected"]:
            difference = ResidualRecording(self.recordings["static"], self.recordings[key])
            self.residuals[key] = np.zeros((self.recordings["static"].get_num_channels(), 0))

            for i in np.arange(t_start * fr, t_stop * fr, fr):
                data = np.linalg.norm(difference.get_traces(start_frame=i, end_frame=i + fr), axis=0) / np.sqrt(fr)
                self.residuals[key] = np.hstack((self.residuals[key], data[:, np.newaxis]))

        return self.residuals, (t_start, t_stop)

    def compute_accuracies(self):
        for case in self.sorter_cases:
            label = case["label"]
            sorting = self.sortings[label]
            if label not in self.comparisons:
                comp = GroundTruthComparison(self.sorting_gt, sorting, exhaustive_gt=True)
                self.comparisons[label] = comp
                self.accuracies[label] = comp.get_performance()["accuracy"].values

    def _plot_accuracy(
        self, accuracies, mode="ordered_accuracy", figsize=(15, 5), axes=None, ax=None, ls="-", legend=True, colors=None
    ):
        if len(self.accuracies) != len(self.sorter_cases):
            self.compute_accuracies()

        n = len(self.sorter_cases)

        if "depth" in mode:
            # gt_unit_positions, _ = mr.extract_units_drift_vector(self.mearec_filenames['drifting'], time_vector=np.array([0., 1.]))
            # unit_depth = gt_unit_positions[0, :]

            template_locations = np.array(mr.load_recordings(self.mearec_filenames["drifting"]).template_locations)
            assert len(template_locations.shape) == 3
            mid = template_locations.shape[1] // 2
            unit_depth = template_locations[:, mid, 2]

            chan_locations = self.recordings["drifting"].get_channel_locations()

        if mode == "ordered_accuracy":
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure

            order = None
            for i, case in enumerate(self.sorter_cases):
                color = colors[i] if colors is not None else None
                label = case["label"]
                # comp = self.comparisons[label]
                acc = accuracies[label]
                order = np.argsort(acc)[::-1]
                acc = acc[order]
                ax.plot(acc, label=label, ls=ls, color=color)
            if legend:
                ax.legend()
            ax.set_ylabel("accuracy")
            ax.set_xlabel("units ordered by accuracy")

        elif mode == "depth_snr":
            if axes is None:
                fig, axs = plt.subplots(nrows=n, figsize=figsize, sharey=True, sharex=True)
            else:
                fig = axes[0].figure
                axs = axes

            metrics = compute_quality_metrics(self.waveforms["static"], metric_names=["snr"], load_if_exists=True)
            snr = metrics["snr"].values

            for i, case in enumerate(self.sorter_cases):
                ax = axs[i]
                label = case["label"]
                acc = accuracies[label]

                points = ax.scatter(unit_depth, snr, c=acc)
                points.set_clim(0.0, 1.0)
                ax.set_title(label)
                ax.axvline(np.min(chan_locations[:, 1]), ls="--", color="k")
                ax.axvline(np.max(chan_locations[:, 1]), ls="--", color="k")
                ax.set_ylabel("snr")
            ax.set_xlabel("depth")

            cbar = fig.colorbar(points, ax=axs[:], location="right", shrink=0.6)
            cbar.ax.set_ylabel("accuracy")

        elif mode == "snr":
            fig, ax = plt.subplots(figsize=figsize)

            metrics = compute_quality_metrics(self.waveforms["static"], metric_names=["snr"], load_if_exists=True)
            snr = metrics["snr"].values

            for i, case in enumerate(self.sorter_cases):
                label = case["label"]
                acc = self.accuracies[label]
                ax.scatter(snr, acc, label=label)
            ax.set_xlabel("snr")
            ax.set_ylabel("accuracy")

            ax.legend()

        elif mode == "depth":
            fig, ax = plt.subplots(figsize=figsize)

            for i, case in enumerate(self.sorter_cases):
                label = case["label"]
                acc = accuracies[label]

                ax.scatter(unit_depth, acc, label=label)
            ax.axvline(np.min(chan_locations[:, 1]), ls="--", color="k")
            ax.axvline(np.max(chan_locations[:, 1]), ls="--", color="k")
            ax.legend()
            ax.set_xlabel("depth")
            ax.set_ylabel("accuracy")

        return fig

    def plot_sortings_accuracy(self, **kwargs):
        if len(self.accuracies) != len(self.sorter_cases):
            self.compute_accuracies()

        return self._plot_accuracy(self.accuracies, ls="-", **kwargs)

    def plot_best_merges_accuracy(self, **kwargs):
        return self._plot_accuracy(self.merged_accuracies, **kwargs, ls="--")

    def plot_sorting_units_categories(self):
        if len(self.accuracies) != len(self.sorter_cases):
            self.compute_accuracies()

        for i, case in enumerate(self.sorter_cases):
            label = case["label"]
            comp = self.comparisons[label]
            count = comp.count_units_categories()
            if i == 0:
                df = pd.DataFrame(columns=count.index)
            df.loc[label, :] = count
        df.plot.bar()

    def find_best_merges(self, merging_score=0.2):
        # this find best merges having the ground truth

        self.merged_sortings = {}
        self.merged_comparisons = {}
        self.merged_accuracies = {}
        self.units_to_merge = {}
        for i, case in enumerate(self.sorter_cases):
            label = case["label"]
            # print()
            # print(label)
            gt_unit_ids = self.sorting_gt.unit_ids
            sorting = self.sortings[label]
            unit_ids = sorting.unit_ids

            comp = self.comparisons[label]
            scores = comp.agreement_scores

            to_merge = []
            for gt_unit_id in gt_unit_ids:
                (inds,) = np.nonzero(scores.loc[gt_unit_id, :].values > merging_score)
                merge_ids = unit_ids[inds]
                if merge_ids.size > 1:
                    to_merge.append(list(merge_ids))

            self.units_to_merge[label] = to_merge
            merged_sporting = MergeUnitsSorting(sorting, to_merge)
            comp_merged = GroundTruthComparison(self.sorting_gt, merged_sporting, exhaustive_gt=True)

            self.merged_sortings[label] = merged_sporting
            self.merged_comparisons[label] = comp_merged
            self.merged_accuracies[label] = comp_merged.get_performance()["accuracy"].values


def plot_distances_to_static(benchmarks, metric="cosine", figsize=(15, 10)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2)

    ax = fig.add_subplot(gs[0:2, 0])
    for count, bench in enumerate(benchmarks):
        distances = bench.compute_distances_to_static(force=False)
        print(distances.keys())
        ax.scatter(
            distances["drifting"][f"template_{metric}"],
            distances["corrected"][f"template_{metric}"],
            c=f"C{count}",
            alpha=0.5,
            label=bench.title,
        )

    ax.legend()

    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [xmin, xmax], "k--")
    _simpleaxis(ax)
    if metric == "euclidean":
        ax.set_xlabel(r"$\|drift - static\|_2$")
        ax.set_ylabel(r"$\|corrected - static\|_2$")
    elif metric == "cosine":
        ax.set_xlabel(r"$cosine(drift, static)$")
        ax.set_ylabel(r"$cosine(corrected, static)$")

    recgen = mr.load_recordings(benchmarks[0].mearec_filenames["static"])
    nb_templates, nb_versions, _ = recgen.template_locations.shape
    template_positions = recgen.template_locations[:, nb_versions // 2, 1:3]
    distances_to_center = template_positions[:, 1]

    ax_1 = fig.add_subplot(gs[0, 1])
    ax_2 = fig.add_subplot(gs[1, 1])
    ax_3 = fig.add_subplot(gs[2:, 1])
    ax_4 = fig.add_subplot(gs[2:, 0])

    for count, bench in enumerate(benchmarks):
        # results = bench._compute_snippets_variability(metric=metric, num_channels=num_channels)
        distances = bench.compute_distances_to_static(force=False)

        m_differences = distances["corrected"][f"wf_{metric}_mean"] / distances["static"][f"wf_{metric}_mean"]
        s_differences = distances["corrected"][f"wf_{metric}_std"] / distances["static"][f"wf_{metric}_std"]

        ax_3.bar([count], [m_differences.mean()], yerr=[m_differences.std()], color=f"C{count}")
        ax_4.bar([count], [s_differences.mean()], yerr=[s_differences.std()], color=f"C{count}")
        idx = np.argsort(distances_to_center)
        ax_1.scatter(distances_to_center[idx], m_differences[idx], color=f"C{count}")
        ax_2.scatter(distances_to_center[idx], s_differences[idx], color=f"C{count}")

    for a in [ax_1, ax_2, ax_3, ax_4]:
        _simpleaxis(a)

    if metric == "euclidean":
        ax_1.set_ylabel(r"$\Delta mean(\|~\|_2)$  (% static)")
        ax_2.set_ylabel(r"$\Delta std(\|~\|_2)$  (% static)")
        ax_3.set_ylabel(r"$\Delta mean(\|~\|_2)$  (% static)")
        ax_4.set_ylabel(r"$\Delta std(\|~\|_2)$  (% static)")
    elif metric == "cosine":
        ax_1.set_ylabel(r"$\Delta mean(cosine)$  (% static)")
        ax_2.set_ylabel(r"$\Delta std(cosine)$  (% static)")
        ax_3.set_ylabel(r"$\Delta mean(cosine)$  (% static)")
        ax_4.set_ylabel(r"$\Delta std(cosine)$  (% static)")
    ax_3.set_xticks(np.arange(len(benchmarks)), [i.title for i in benchmarks])
    ax_4.set_xticks(np.arange(len(benchmarks)), [i.title for i in benchmarks])
    xmin, xmax = ax_3.get_xlim()
    ax_3.plot([xmin, xmax], [1, 1], "k--")
    ax_4.plot([xmin, xmax], [1, 1], "k--")
    ax_1.set_xticks([])
    ax_2.set_xlabel("depth (um)")

    xmin, xmax = ax_1.get_xlim()
    ax_1.plot([xmin, xmax], [1, 1], "k--")
    ax_2.plot([xmin, xmax], [1, 1], "k--")
    plt.tight_layout()


def plot_snr_decrease(benchmarks, figsize=(15, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize, squeeze=False)

    recgen = mr.load_recordings(benchmarks[0].mearec_filenames["static"])
    nb_templates, nb_versions, _ = recgen.template_locations.shape
    template_positions = recgen.template_locations[:, nb_versions // 2, 1:3]
    distances_to_center = template_positions[:, 1]
    idx = np.argsort(distances_to_center)
    _simpleaxis(axes[0, 0])

    snr_static = compute_quality_metrics(benchmarks[0].waveforms["static"], metric_names=["snr"], load_if_exists=True)
    snr_drifting = compute_quality_metrics(
        benchmarks[0].waveforms["drifting"], metric_names=["snr"], load_if_exists=True
    )

    m = np.max(snr_static)
    axes[0, 0].scatter(snr_static.values, snr_drifting.values, c="0.5")
    axes[0, 0].plot([0, m], [0, m], color="k")

    axes[0, 0].set_ylabel("units SNR for drifting")
    _simpleaxis(axes[0, 0])

    axes[0, 1].plot(distances_to_center[idx], (snr_drifting.values / snr_static.values)[idx], c="0.5")
    axes[0, 1].plot(distances_to_center[idx], np.ones(len(idx)), "k--")
    _simpleaxis(axes[0, 1])
    axes[0, 1].set_xticks([])
    axes[0, 0].set_xticks([])

    for count, bench in enumerate(benchmarks):
        snr_corrected = compute_quality_metrics(bench.waveforms["corrected"], metric_names=["snr"], load_if_exists=True)
        axes[1, 0].scatter(snr_static.values, snr_corrected.values, label=bench.title)
        axes[1, 0].plot([0, m], [0, m], color="k")

        axes[1, 1].plot(distances_to_center[idx], (snr_corrected.values / snr_static.values)[idx], c=f"C{count}")

    axes[1, 0].set_xlabel("units SNR for static")
    axes[1, 0].set_ylabel("units SNR for corrected")
    axes[1, 1].plot(distances_to_center[idx], np.ones(len(idx)), "k--")
    axes[1, 0].legend()
    _simpleaxis(axes[1, 0])
    _simpleaxis(axes[1, 1])
    axes[1, 1].set_ylabel(r"$\Delta(SNR)$")
    axes[0, 1].set_ylabel(r"$\Delta(SNR)$")

    axes[1, 1].set_xlabel("depth (um)")


def plot_residuals_comparisons(benchmarks):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for count, bench in enumerate(benchmarks):
        residuals, (t_start, t_stop) = bench.compute_residuals(force=False)
        time_axis = np.arange(t_start, t_stop)
        axes[0].plot(time_axis, residuals["corrected"].mean(0), label=bench.title)
    axes[0].legend()
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel(r"$|S_{corrected} - S_{static}|$")
    _simpleaxis(axes[0])

    channel_positions = benchmarks[0].recordings["static"].get_channel_locations()
    distances_to_center = channel_positions[:, 1]
    idx = np.argsort(distances_to_center)

    for count, bench in enumerate(benchmarks):
        residuals, (t_start, t_stop) = bench.compute_residuals(force=False)
        time_axis = np.arange(t_start, t_stop)
        axes[1].plot(
            distances_to_center[idx], residuals["corrected"].mean(1)[idx], label=bench.title, lw=2, c=f"C{count}"
        )
        axes[1].fill_between(
            distances_to_center[idx],
            residuals["corrected"].mean(1)[idx] - residuals["corrected"].std(1)[idx],
            residuals["corrected"].mean(1)[idx] + residuals["corrected"].std(1)[idx],
            color=f"C{count}",
            alpha=0.25,
        )
    axes[1].set_xlabel("depth (um)")
    _simpleaxis(axes[1])

    for count, bench in enumerate(benchmarks):
        residuals, (t_start, t_stop) = bench.compute_residuals(force=False)
        axes[2].bar([count], [residuals["corrected"].mean()], yerr=[residuals["corrected"].std()], color=f"C{count}")

    _simpleaxis(axes[2])
    axes[2].set_xticks(np.arange(len(benchmarks)), [i.title for i in benchmarks])


from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class ResidualRecording(BasePreprocessor):
    name = "residual_recording"

    def __init__(self, recording_1, recording_2):
        assert recording_1.get_num_segments() == recording_2.get_num_segments()
        BasePreprocessor.__init__(self, recording_1)

        for parent_recording_segment_1, parent_recording_segment_2 in zip(
            recording_1._recording_segments, recording_2._recording_segments
        ):
            rec_segment = DifferenceRecordingSegment(parent_recording_segment_1, parent_recording_segment_2)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording_1=recording_1, recording_2=recording_2)


class DifferenceRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment_1, parent_recording_segment_2):
        BasePreprocessorSegment.__init__(self, parent_recording_segment_1)
        self.parent_recording_segment_1 = parent_recording_segment_1
        self.parent_recording_segment_2 = parent_recording_segment_2

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces_1 = self.parent_recording_segment_1.get_traces(start_frame, end_frame, channel_indices)
        traces_2 = self.parent_recording_segment_2.get_traces(start_frame, end_frame, channel_indices)

        return traces_2 - traces_1


colors = {"static": "C0", "drifting": "C1", "corrected": "C2"}
