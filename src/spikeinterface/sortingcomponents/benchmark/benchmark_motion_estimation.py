from __future__ import annotations

import json
import numpy as np
import time
from pathlib import Path


from spikeinterface.core import get_noise_levels
from spikeinterface.extractors import read_mearec
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_interpolation import correct_motion_on_peaks
from spikeinterface.preprocessing import bandpass_filter, zscore, common_reference

from spikeinterface.sortingcomponents.benchmark.benchmark_tools import BenchmarkBase, _simpleaxis


from spikeinterface.widgets import plot_probe_map

import scipy.interpolate

import matplotlib.pyplot as plt

import MEArec as mr


class BenchmarkMotionEstimationMearec(BenchmarkBase):
    _array_names = (
        "noise_levels",
        "gt_unit_positions",
        "peaks",
        "selected_peaks",
        "motion",
        "temporal_bins",
        "spatial_bins",
        "peak_locations",
        "gt_motion",
    )

    def __init__(
        self,
        mearec_filename,
        title="",
        detect_kwargs={},
        select_kwargs=None,
        localize_kwargs={},
        estimate_motion_kwargs={},
        folder=None,
        do_preprocessing=True,
        job_kwargs={"chunk_duration": "1s", "n_jobs": -1, "progress_bar": True, "verbose": True},
        overwrite=False,
        parent_benchmark=None,
    ):
        BenchmarkBase.__init__(
            self, folder=folder, title=title, overwrite=overwrite, job_kwargs=job_kwargs, parent_benchmark=None
        )

        self._args.extend([str(mearec_filename)])

        self.mearec_filename = mearec_filename
        self.raw_recording, self.gt_sorting = read_mearec(self.mearec_filename)
        self.do_preprocessing = do_preprocessing

        self._recording = None
        self.detect_kwargs = detect_kwargs.copy()
        self.select_kwargs = select_kwargs.copy() if select_kwargs is not None else None
        self.localize_kwargs = localize_kwargs.copy()
        self.estimate_motion_kwargs = estimate_motion_kwargs.copy()

        self._kwargs.update(
            dict(
                detect_kwargs=self.detect_kwargs,
                select_kwargs=self.select_kwargs,
                localize_kwargs=self.localize_kwargs,
                estimate_motion_kwargs=self.estimate_motion_kwargs,
            )
        )

    @property
    def recording(self):
        if self._recording is None:
            if self.do_preprocessing:
                self._recording = bandpass_filter(self.raw_recording)
                self._recording = common_reference(self._recording)
                self._recording = zscore(self._recording)
            else:
                self._recording = self.raw_recording
        return self._recording

    def run(self):
        if self.folder is not None:
            if self.folder.exists() and not self.overwrite:
                raise ValueError(f"The folder {self.folder} is not empty")

        self.noise_levels = get_noise_levels(self.recording, return_scaled=False)

        t0 = time.perf_counter()
        self.peaks = detect_peaks(
            self.recording, noise_levels=self.noise_levels, **self.detect_kwargs, **self.job_kwargs
        )
        t1 = time.perf_counter()
        if self.select_kwargs is not None:
            self.selected_peaks = select_peaks(self.peaks, **self.select_kwargs, **self.job_kwargs)
        else:
            self.selected_peaks = self.peaks
        t2 = time.perf_counter()
        self.peak_locations = localize_peaks(
            self.recording, self.selected_peaks, **self.localize_kwargs, **self.job_kwargs
        )
        t3 = time.perf_counter()
        self.motion, self.temporal_bins, self.spatial_bins = estimate_motion(
            self.recording, self.selected_peaks, self.peak_locations, **self.estimate_motion_kwargs
        )

        t4 = time.perf_counter()

        self.run_times = dict(
            detect_peaks=t1 - t0,
            select_peaks=t2 - t1,
            localize_peaks=t3 - t2,
            estimate_motion=t4 - t3,
        )

        self.compute_gt_motion()

        # align globally gt_motion and motion to avoid offsets
        self.motion += np.median(self.gt_motion - self.motion)

        ## save folder
        if self.folder is not None:
            self.save_to_folder()

    def run_estimate_motion(self):
        # usefull to re run only the motion estimate with peak localization
        t3 = time.perf_counter()
        self.motion, self.temporal_bins, self.spatial_bins = estimate_motion(
            self.recording, self.selected_peaks, self.peak_locations, **self.estimate_motion_kwargs
        )
        t4 = time.perf_counter()

        self.compute_gt_motion()

        # align globally gt_motion and motion to avoid offsets
        self.motion += np.median(self.gt_motion - self.motion)
        self.run_times["estimate_motion"] = t4 - t3

        ## save folder
        if self.folder is not None:
            self.save_to_folder()

    def compute_gt_motion(self):
        self.gt_unit_positions, _ = mr.extract_units_drift_vector(self.mearec_filename, time_vector=self.temporal_bins)

        template_locations = np.array(mr.load_recordings(self.mearec_filename).template_locations)
        assert len(template_locations.shape) == 3
        mid = template_locations.shape[1] // 2
        unit_mid_positions = template_locations[:, mid, 2]

        unit_motions = self.gt_unit_positions - unit_mid_positions
        # unit_positions = np.mean(self.gt_unit_positions, axis=0)

        if self.spatial_bins is None:
            self.gt_motion = np.mean(unit_motions, axis=1)[:, None]
            channel_positions = self.recording.get_channel_locations()
            probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
            center = (probe_y_min + probe_y_max) // 2
            self.spatial_bins = np.array([center])
        else:
            # time, units
            self.gt_motion = np.zeros_like(self.motion)
            for t in range(self.gt_unit_positions.shape[0]):
                f = scipy.interpolate.interp1d(unit_mid_positions, unit_motions[t, :], fill_value="extrapolate")
                self.gt_motion[t, :] = f(self.spatial_bins)

    def plot_true_drift(self, scaling_probe=1.5, figsize=(15, 10), axes=None):
        if axes is None:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 8, wspace=0)

        if axes is None:
            ax = fig.add_subplot(gs[:2])
        else:
            ax = axes[0]
        plot_probe_map(self.recording, ax=ax)
        _simpleaxis(ax)

        mr_recording = mr.load_recordings(self.mearec_filename)

        for loc in mr_recording.template_locations[::2]:
            if len(mr_recording.template_locations.shape) == 3:
                ax.plot([loc[0, 1], loc[-1, 1]], [loc[0, 2], loc[-1, 2]], alpha=0.7, lw=2)
            else:
                ax.scatter([loc[1]], [loc[2]], alpha=0.7, s=100)

        # ymin, ymax = ax.get_ylim()
        ax.set_ylabel("depth (um)")
        ax.set_xlabel(None)
        # ax.set_yticks(np.arange(-600,600,100), np.arange(-600,600,100))

        # ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)
        if axes is None:
            ax = fig.add_subplot(gs[2:7])
        else:
            ax = axes[1]

        for i in range(self.gt_unit_positions.shape[1]):
            ax.plot(self.temporal_bins, self.gt_unit_positions[:, i], alpha=0.5, ls="--", c="0.5")

        for i in range(self.gt_motion.shape[1]):
            depth = self.spatial_bins[i]
            ax.plot(self.temporal_bins, self.gt_motion[:, i] + depth, color="green", lw=4)

        # ax.set_ylim(ymin, ymax)
        ax.set_xlabel("time (s)")
        _simpleaxis(ax)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

        channel_positions = self.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
        ax.set_ylim(scaling_probe * probe_y_min, scaling_probe * probe_y_max)

        ax.axhline(probe_y_min, color="k", ls="--", alpha=0.5)
        ax.axhline(probe_y_max, color="k", ls="--", alpha=0.5)

        if axes is None:
            ax = fig.add_subplot(gs[7])
        else:
            ax = axes[2]
        # plot_probe_map(self.recording, ax=ax)
        _simpleaxis(ax)

        ax.hist(self.gt_unit_positions[30, :], 50, orientation="horizontal", color="0.5")
        ax.set_yticks([])
        ax.set_xlabel("# neurons")

    def plot_peaks_probe(self, alpha=0.05, figsize=(15, 10)):
        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=figsize)
        ax = axs[0]
        plot_probe_map(self.recording, ax=ax)
        ax.scatter(self.peak_locations["x"], self.peak_locations["y"], color="k", s=1, alpha=alpha)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if "z" in self.peak_locations.dtype.fields:
            ax = axs[1]
            ax.scatter(self.peak_locations["z"], self.peak_locations["y"], color="k", s=1, alpha=alpha)
            ax.set_xlabel("z")
            ax.set_xlim(0, 100)

    def plot_peaks(self, scaling_probe=1.5, show_drift=True, show_histogram=True, alpha=0.05, figsize=(15, 10)):
        fig = plt.figure(figsize=figsize)
        if show_histogram:
            gs = fig.add_gridspec(1, 4)
        else:
            gs = fig.add_gridspec(1, 3)
        # Create the Axes.

        ax0 = fig.add_subplot(gs[0])
        plot_probe_map(self.recording, ax=ax0)
        _simpleaxis(ax0)

        # ymin, ymax = ax.get_ylim()
        ax0.set_ylabel("depth (um)")
        ax0.set_xlabel(None)

        ax = ax1 = fig.add_subplot(gs[1:3])
        x = self.selected_peaks["sample_index"] / self.recording.get_sampling_frequency()
        y = self.peak_locations["y"]
        ax.scatter(x, y, s=1, color="k", alpha=alpha)

        ax.set_title(self.title)
        # xmin, xmax = ax.get_xlim()
        # ax.plot([xmin, xmax], [probe_y_min, probe_y_min], 'k--', alpha=0.5)
        # ax.plot([xmin, xmax], [probe_y_max, probe_y_max], 'k--', alpha=0.5)

        _simpleaxis(ax)
        # ax.set_yticks([])
        # ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("time (s)")

        channel_positions = self.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
        ax.set_ylim(scaling_probe * probe_y_min, scaling_probe * probe_y_max)

        ax.axhline(probe_y_min, color="k", ls="--", alpha=0.5)
        ax.axhline(probe_y_max, color="k", ls="--", alpha=0.5)

        if show_drift:
            if self.spatial_bins is None:
                center = (probe_y_min + probe_y_max) // 2
                ax.plot(self.temporal_bins, self.gt_motion[:, 0] + center, color="green", lw=1.5)
                ax.plot(self.temporal_bins, self.motion[:, 0] + center, color="orange", lw=1.5)
            else:
                for i in range(self.gt_motion.shape[1]):
                    depth = self.spatial_bins[i]
                    ax.plot(self.temporal_bins, self.gt_motion[:, i] + depth, color="green", lw=1.5)
                    ax.plot(self.temporal_bins, self.motion[:, i] + depth, color="orange", lw=1.5)

        if show_histogram:
            ax2 = fig.add_subplot(gs[3])
            ax2.hist(self.peak_locations["y"], bins=1000, orientation="horizontal")

            ax2.axhline(probe_y_min, color="k", ls="--", alpha=0.5)
            ax2.axhline(probe_y_max, color="k", ls="--", alpha=0.5)

            ax2.set_xlabel("density")
            _simpleaxis(ax2)
            # ax.set_ylabel('')
            ax.set_yticks([])
            ax2.sharey(ax0)

        ax1.sharey(ax0)

    def plot_motion_corrected_peaks(self, scaling_probe=1.5, alpha=0.05, figsize=(15, 10), show_probe=True, axes=None):
        if axes is None:
            fig = plt.figure(figsize=figsize)
            if show_probe:
                gs = fig.add_gridspec(1, 5)
            else:
                gs = fig.add_gridspec(1, 4)
        # Create the Axes.

        if show_probe:
            if axes is None:
                ax0 = ax = fig.add_subplot(gs[0])
            else:
                ax0 = ax = axes[0]
            plot_probe_map(self.recording, ax=ax)
            _simpleaxis(ax)

            ymin, ymax = ax.get_ylim()
            ax.set_ylabel("depth (um)")
            ax.set_xlabel(None)

        channel_positions = self.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()

        peak_locations_corrected = correct_motion_on_peaks(
            self.selected_peaks,
            self.peak_locations,
            self.recording.sampling_frequency,
            self.motion,
            self.temporal_bins,
            self.spatial_bins,
            direction="y",
        )
        if axes is None:
            if show_probe:
                ax1 = ax = fig.add_subplot(gs[1:3])
            else:
                ax1 = ax = fig.add_subplot(gs[0:2])
        else:
            if show_probe:
                ax1 = ax = axes[1]
            else:
                ax1 = ax = axes[0]

        _simpleaxis(ax)

        x = self.selected_peaks["sample_index"] / self.recording.get_sampling_frequency()
        y = self.peak_locations["y"]
        ax.scatter(x, y, s=1, color="k", alpha=alpha)
        ax.set_title(self.title)

        ax.axhline(probe_y_min, color="k", ls="--", alpha=0.5)
        ax.axhline(probe_y_max, color="k", ls="--", alpha=0.5)

        ax.set_xlabel("time (s)")

        if axes is None:
            if show_probe:
                ax2 = ax = fig.add_subplot(gs[3:5])
            else:
                ax2 = ax = fig.add_subplot(gs[2:4])
        else:
            if show_probe:
                ax2 = ax = axes[2]
            else:
                ax2 = ax = axes[1]

        _simpleaxis(ax)
        y = peak_locations_corrected["y"]
        ax.scatter(x, y, s=1, color="k", alpha=alpha)

        ax.axhline(probe_y_min, color="k", ls="--", alpha=0.5)
        ax.axhline(probe_y_max, color="k", ls="--", alpha=0.5)

        ax.set_xlabel("time (s)")

        if show_probe:
            ax0.set_ylim(scaling_probe * probe_y_min, scaling_probe * probe_y_max)
            ax1.sharey(ax0)
            ax2.sharey(ax0)
        else:
            ax1.set_ylim(scaling_probe * probe_y_min, scaling_probe * probe_y_max)
            ax2.sharey(ax1)

    def estimation_vs_depth(self, show_only=8, figsize=(15, 10)):
        fig, axs = plt.subplots(ncols=2, figsize=figsize, sharey=True)

        n = self.motion.shape[1]
        step = int(np.ceil(max(1, n / show_only)))
        colors = plt.cm.get_cmap("jet", n)
        for i in range(0, n, step):
            ax = axs[0]
            ax.plot(self.temporal_bins, self.gt_motion[:, i], lw=1.5, ls="--", color=colors(i))
            ax.plot(
                self.temporal_bins,
                self.motion[:, i],
                lw=1.5,
                ls="-",
                color=colors(i),
                label=f"{self.spatial_bins[i]:0.1f}",
            )

            ax = axs[1]
            ax.plot(self.temporal_bins, self.motion[:, i] - self.gt_motion[:, i], lw=1.5, ls="-", color=colors(i))

        ax = axs[0]
        ax.set_title(self.title)
        ax.legend()
        ax.set_ylabel("drift estimated and GT(um)")
        ax.set_xlabel("time (s)")
        _simpleaxis(ax)

        ax = axs[1]
        ax.set_ylabel("error (um)")
        ax.set_xlabel("time (s)")
        _simpleaxis(ax)

    def view_errors(self, figsize=(15, 10), lim=None):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2)

        errors = self.gt_motion - self.motion

        channel_positions = self.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()

        ax = fig.add_subplot(gs[0, :])
        im = ax.imshow(
            np.abs(errors).T,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=(self.temporal_bins[0], self.temporal_bins[-1], self.spatial_bins[0], self.spatial_bins[-1]),
        )
        plt.colorbar(im, ax=ax, label="error")
        ax.set_ylabel("depth (um)")
        ax.set_xlabel("time (s)")
        ax.set_title(self.title)
        if lim is not None:
            im.set_clim(0, lim)

        ax = fig.add_subplot(gs[1, 0])
        mean_error = np.sqrt(np.mean((errors) ** 2, axis=1))
        ax.plot(self.temporal_bins, mean_error)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("error")
        _simpleaxis(ax)
        if lim is not None:
            ax.set_ylim(0, lim)

        ax = fig.add_subplot(gs[1, 1])
        depth_error = np.sqrt(np.mean((errors) ** 2, axis=0))
        ax.plot(self.spatial_bins, depth_error)
        ax.axvline(probe_y_min, color="k", ls="--", alpha=0.5)
        ax.axvline(probe_y_max, color="k", ls="--", alpha=0.5)
        ax.set_xlabel("depth (um)")
        ax.set_ylabel("error")
        _simpleaxis(ax)
        if lim is not None:
            ax.set_ylim(0, lim)

        return fig


def plot_errors_several_benchmarks(benchmarks, axes=None, show_legend=True, colors=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for count, benchmark in enumerate(benchmarks):
        c = colors[count] if colors is not None else None
        errors = benchmark.gt_motion - benchmark.motion
        mean_error = np.sqrt(np.mean((errors) ** 2, axis=1))
        depth_error = np.sqrt(np.mean((errors) ** 2, axis=0))

        axes[0].plot(benchmark.temporal_bins, mean_error, lw=1, label=benchmark.title, color=c)
        parts = axes[1].violinplot(mean_error, [count], showmeans=True)
        if c is not None:
            for pc in parts["bodies"]:
                pc.set_facecolor(c)
                pc.set_edgecolor(c)
            for k in parts:
                if k != "bodies":
                    # for line in parts[k]:
                    parts[k].set_color(c)
        axes[2].plot(benchmark.spatial_bins, depth_error, label=benchmark.title, color=c)

    ax0 = ax = axes[0]
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [μm]")
    if show_legend:
        ax.legend()
    _simpleaxis(ax)

    ax1 = axes[1]
    # ax.set_ylabel('error')
    ax1.set_yticks([])
    ax1.set_xticks([])
    _simpleaxis(ax1)

    ax2 = axes[2]
    ax2.set_yticks([])
    ax2.set_xlabel("Depth [μm]")
    # ax.set_ylabel('error')
    channel_positions = benchmark.recording.get_channel_locations()
    probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
    ax2.axvline(probe_y_min, color="k", ls="--", alpha=0.5)
    ax2.axvline(probe_y_max, color="k", ls="--", alpha=0.5)

    _simpleaxis(ax2)

    # ax1.sharey(ax0)
    # ax2.sharey(ax0)


def plot_error_map_several_benchmarks(benchmarks, axes=None, lim=15, figsize=(10, 10)):
    if axes is None:
        fig, axes = plt.subplots(nrows=len(benchmarks), sharex=True, sharey=True, figsize=figsize)
    else:
        fig = axes[0].figure

    for count, benchmark in enumerate(benchmarks):
        errors = benchmark.gt_motion - benchmark.motion

        channel_positions = benchmark.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()

        ax = axes[count]
        im = ax.imshow(
            np.abs(errors).T,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=(
                benchmark.temporal_bins[0],
                benchmark.temporal_bins[-1],
                benchmark.spatial_bins[0],
                benchmark.spatial_bins[-1],
            ),
        )
        fig.colorbar(im, ax=ax, label="error")
        ax.set_ylabel("depth (um)")

        ax.set_title(benchmark.title)
        if lim is not None:
            im.set_clim(0, lim)

    axes[-1].set_xlabel("time (s)")

    return fig


def plot_motions_several_benchmarks(benchmarks):
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(list(benchmarks)[0].temporal_bins, list(benchmarks)[0].gt_motion[:, 0], lw=2, c="k", label="real motion")
    for count, benchmark in enumerate(benchmarks):
        ax.plot(benchmark.temporal_bins, benchmark.motion.mean(1), lw=1, c=f"C{count}", label=benchmark.title)
        ax.fill_between(
            benchmark.temporal_bins,
            benchmark.motion.mean(1) - benchmark.motion.std(1),
            benchmark.motion.mean(1) + benchmark.motion.std(1),
            color=f"C{count}",
            alpha=0.25,
        )

    # ax.legend()
    ax.set_ylabel("depth (um)")
    ax.set_xlabel("time (s)")
    _simpleaxis(ax)


def plot_speed_several_benchmarks(benchmarks, detailed=True, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    for count, benchmark in enumerate(benchmarks):
        color = colors[count] if colors is not None else None

        if detailed:
            bottom = 0
            i = 0
            patterns = ["/", "\\", "|", "*"]
            for key, value in benchmark.run_times.items():
                if count == 0:
                    label = key.replace("_", " ")
                else:
                    label = None
                ax.bar([count], [value], label=label, bottom=bottom, color=color, edgecolor="black", hatch=patterns[i])
                bottom += value
                i += 1
        else:
            total_run_time = np.sum([value for key, value in benchmark.run_times.items()])
            ax.bar([count], [total_run_time], color=color, edgecolor="black")

    # ax.legend()
    ax.set_ylabel("speed (s)")
    _simpleaxis(ax)
    ax.set_xticks([])
    # ax.set_xticks(np.arange(len(benchmarks)), [i.title for i in benchmarks])
