from __future__ import annotations

from spikeinterface.core import extract_waveforms
from spikeinterface.preprocessing import bandpass_filter, common_reference
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.qualitymetrics import compute_quality_metrics, compute_snrs
from spikeinterface.widgets import (
    plot_probe_map,
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
    plot_unit_templates,
    plot_unit_waveforms,
)
from spikeinterface.postprocessing import compute_spike_locations
from spikeinterface.postprocessing.unit_localization import compute_center_of_mass, compute_monopolar_triangulation
from spikeinterface.core import get_noise_levels
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import BenchmarkBase, _simpleaxis

import time
import string, random
import pylab as plt
import os
import numpy as np


class BenchmarkPeakLocalization:
    def __init__(self, recording, gt_sorting, gt_positions, job_kwargs={}, tmp_folder=None, verbose=True, title=None):
        self.verbose = verbose
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.job_kwargs = job_kwargs
        self.sampling_rate = self.recording.get_sampling_frequency()
        self.title = title
        self.waveforms = None

        self.tmp_folder = tmp_folder
        if self.tmp_folder is None:
            self.tmp_folder = os.path.join(".", "".join(random.choices(string.ascii_uppercase + string.digits, k=8)))

        self.gt_positions = gt_positions

    def __del__(self):
        import shutil

        shutil.rmtree(self.tmp_folder)

    def run(self, method, method_kwargs={}):
        if self.waveforms is None:
            self.waveforms = extract_waveforms(
                self.recording,
                self.gt_sorting,
                self.tmp_folder,
                ms_before=2.5,
                ms_after=2.5,
                max_spikes_per_unit=500,
                return_scaled=False,
                **self.job_kwargs,
            )

        t_start = time.time()
        if self.title is None:
            self.title = method

        unit_params = method_kwargs.copy()

        for key in ["ms_after", "ms_before"]:
            if key in unit_params:
                unit_params.pop(key)

        if method == "center_of_mass":
            self.template_positions = compute_center_of_mass(self.waveforms, **unit_params)
        elif method == "monopolar_triangulation":
            self.template_positions = compute_monopolar_triangulation(self.waveforms, **unit_params)

        self.spike_positions = compute_spike_locations(
            self.waveforms, method=method, method_kwargs=method_kwargs, **self.job_kwargs, outputs="by_unit"
        )

        self.raw_templates_results = {}

        for unit_ind, unit_id in enumerate(self.waveforms.sorting.unit_ids):
            data = self.spike_positions[0][unit_id]
            self.raw_templates_results[unit_id] = np.sqrt(
                (data["x"] - self.gt_positions[unit_ind, 0]) ** 2 + (data["y"] - self.gt_positions[unit_ind, 1]) ** 2
            )

        self.medians_over_templates = np.array(
            [np.median(self.raw_templates_results[unit_id]) for unit_id in self.waveforms.sorting.unit_ids]
        )
        self.mads_over_templates = np.array(
            [
                np.median(np.abs(self.raw_templates_results[unit_id] - np.median(self.raw_templates_results[unit_id])))
                for unit_id in self.waveforms.sorting.unit_ids
            ]
        )

    def plot_template_errors(self, show_probe=True):
        import spikeinterface.full as si
        import pylab as plt

        si.plot_probe_map(self.recording)
        plt.scatter(self.gt_positions[:, 0], self.gt_positions[:, 1], c=np.arange(len(self.gt_positions)), cmap="jet")
        plt.scatter(
            self.template_positions[:, 0],
            self.template_positions[:, 1],
            c=np.arange(len(self.template_positions)),
            cmap="jet",
            marker="v",
        )


def plot_comparison_positions(benchmarks, mode="average"):
    norms = np.linalg.norm(benchmarks[0].waveforms.get_all_templates(mode=mode), axis=(1, 2))
    distances_to_center = np.linalg.norm(benchmarks[0].gt_positions[:, :2], axis=1)
    zdx = np.argsort(distances_to_center)
    idx = np.argsort(norms)

    snrs_tmp = compute_snrs(benchmarks[0].waveforms)
    snrs = np.zeros(len(snrs_tmp))
    for k, v in snrs_tmp.items():
        snrs[int(k[1:])] = v

    wdx = np.argsort(snrs)

    plt.rc("font", size=13)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
    ax = axs[0, 0]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_title(title)

    from scipy.signal import savgol_filter

    smoothing_factor = 31

    for bench in benchmarks:
        errors = np.linalg.norm(bench.template_positions[:, :2] - bench.gt_positions[:, :2], axis=1)
        ax.plot(snrs[wdx], savgol_filter(errors[wdx], smoothing_factor, 3), label=bench.title)

    ax.legend()
    # ax.set_xlabel('norm')
    ax.set_ylabel("error (um)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)

    ax = axs[0, 1]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bench in benchmarks:
        errors = np.linalg.norm(bench.template_positions[:, :2] - bench.gt_positions[:, :2], axis=1)
        ax.plot(distances_to_center[zdx], savgol_filter(errors[zdx], smoothing_factor, 3), label=bench.title)

    # ax.set_xlabel('distance to center (um)')
    # ax.set_yticks([])

    ax = axs[0, 2]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_title(title)

    for count, bench in enumerate(benchmarks):
        errors = np.linalg.norm(bench.template_positions[:, :2] - bench.gt_positions[:, :2], axis=1)
        ax.bar([count], np.mean(errors), yerr=np.std(errors))

    # ax.set_xlabel('norms')
    # ax.set_yticks([])
    # ax.set_ylim(ymin, ymax)

    ax = axs[1, 0]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bench in benchmarks:
        ax.plot(
            snrs[wdx], savgol_filter(bench.medians_over_templates[wdx], smoothing_factor, 3), lw=2, label=bench.title
        )
        ymin = savgol_filter((bench.medians_over_templates - bench.mads_over_templates)[wdx], smoothing_factor, 3)
        ymax = savgol_filter((bench.medians_over_templates + bench.mads_over_templates)[wdx], smoothing_factor, 3)

        ax.fill_between(snrs[wdx], ymin, ymax, alpha=0.5)

    ax.set_xlabel("snr")
    ax.set_ylabel("error (um)")
    # ymin, ymax = ax.get_ylim()
    # ax.set_ylim(0, ymax)
    # ax.set_yscale('log')

    ax = axs[1, 1]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bench in benchmarks:
        ax.plot(
            distances_to_center[zdx],
            savgol_filter(bench.medians_over_templates[zdx], smoothing_factor, 3),
            lw=2,
            label=bench.title,
        )
        ymin = savgol_filter((bench.medians_over_templates - bench.mads_over_templates)[zdx], smoothing_factor, 3)
        ymax = savgol_filter((bench.medians_over_templates + bench.mads_over_templates)[zdx], smoothing_factor, 3)

        ax.fill_between(distances_to_center[zdx], ymin, ymax, alpha=0.5)

    ax.set_xlabel("distance to center (um)")

    x_means = []
    x_stds = []
    for count, bench in enumerate(benchmarks):
        x_means += [np.median(bench.medians_over_templates)]
        x_stds += [np.std(bench.medians_over_templates)]

    # ax.set_yticks([])
    # ax.set_ylim(ymin, ymax)

    ax = axs[1, 2]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_means = []
    y_stds = []
    for count, bench in enumerate(benchmarks):
        y_means += [np.median(bench.mads_over_templates)]
        y_stds += [np.std(bench.mads_over_templates)]

    colors = [f"C{i}" for i in range(len(x_means))]
    ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt=".", c="0.5", alpha=0.5)
    ax.scatter(x_means, y_means, c=colors, s=200)

    ax.set_ylabel("error mads (um)")
    ax.set_xlabel("error medians (um)")
    # ax.set_yticks([]
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, 12)


def plot_comparison_inferences(benchmarks, bin_size=np.arange(0.1, 20, 1)):
    import numpy as np
    import sklearn
    import scipy.stats
    import spikeinterface.full as si

    plt.rc("font", size=11)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    from scipy.signal import savgol_filter

    smoothing_factor = 5

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(8, 10)

    ax3 = fig.add_subplot(gs[0:2, 6:10])
    ax4 = fig.add_subplot(gs[2:4, 6:10])

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_ylabel("correlation coefficient")
    ax3.set_xticks([])

    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_ylabel("chi squared")
    ax4.set_xlabel("bin size (um)")

    def chiSquared(p, q):
        return 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-6))

    for count, benchmark in enumerate(benchmarks):
        spikes = benchmark.spike_positions[0]
        units = benchmark.waveforms.sorting.unit_ids
        all_x = np.concatenate([spikes[unit_id]["x"] for unit_id in units])
        all_y = np.concatenate([spikes[unit_id]["y"] for unit_id in units])

        gt_positions = benchmark.gt_positions[:, :2]
        real_x = np.concatenate([gt_positions[c, 0] * np.ones(len(spikes[i]["x"])) for c, i in enumerate(units)])
        real_y = np.concatenate([gt_positions[c, 1] * np.ones(len(spikes[i]["x"])) for c, i in enumerate(units)])

        r_y = np.zeros(len(bin_size))
        c_y = np.zeros(len(bin_size))
        for i, b in enumerate(bin_size):
            all_bins = np.arange(all_y.min(), all_y.max(), b)
            x1, y2 = np.histogram(all_y, bins=all_bins)
            x2, y2 = np.histogram(real_y, bins=all_bins)

            r_y[i] = np.corrcoef(x1, x2)[0, 1]
            c_y[i] = chiSquared(x1, x2)

        r_x = np.zeros(len(bin_size))
        c_x = np.zeros(len(bin_size))
        for i, b in enumerate(bin_size):
            all_bins = np.arange(all_x.min(), all_x.max(), b)
            x1, y2 = np.histogram(all_x, bins=all_bins)
            x2, y2 = np.histogram(real_x, bins=all_bins)

            r_x[i] = np.corrcoef(x1, x2)[0, 1]
            c_x[i] = chiSquared(x1, x2)

        ax3.plot(bin_size, savgol_filter((r_y + r_x) / 2, smoothing_factor, 3), c=f"C{count}", label=benchmark.title)
        ax4.plot(bin_size, savgol_filter((c_y + c_x) / 2, smoothing_factor, 3), c=f"C{count}", label=benchmark.title)

    r_control_y = np.zeros(len(bin_size))
    c_control_y = np.zeros(len(bin_size))
    for i, b in enumerate(bin_size):
        all_bins = np.arange(all_y.min(), all_y.max(), b)
        random_y = all_y.min() + (all_y.max() - all_y.min()) * np.random.rand(len(all_y))
        x1, y2 = np.histogram(random_y, bins=all_bins)
        x2, y2 = np.histogram(real_y, bins=all_bins)

        r_control_y[i] = np.corrcoef(x1, x2)[0, 1]
        c_control_y[i] = chiSquared(x1, x2)

    r_control_x = np.zeros(len(bin_size))
    c_control_x = np.zeros(len(bin_size))
    for i, b in enumerate(bin_size):
        all_bins = np.arange(all_x.min(), all_x.max(), b)
        random_x = all_x.min() + (all_x.max() - all_x.min()) * np.random.rand(len(all_y))
        x1, y2 = np.histogram(random_x, bins=all_bins)
        x2, y2 = np.histogram(real_x, bins=all_bins)

        r_control_x[i] = np.corrcoef(x1, x2)[0, 1]
        c_control_x[i] = chiSquared(x1, x2)

    ax3.plot(bin_size, savgol_filter((r_control_y + r_control_x) / 2, smoothing_factor, 3), "0.5", label="Control")
    ax4.plot(bin_size, savgol_filter((c_control_y + c_control_x) / 2, smoothing_factor, 3), "0.5", label="Control")

    ax4.legend()

    ax0 = fig.add_subplot(gs[0:3, 0:3])

    si.plot_probe_map(benchmarks[0].recording, ax=ax0)
    ax0.scatter(all_x, all_y, alpha=0.5)
    ax0.scatter(gt_positions[:, 0], gt_positions[:, 1], c="k")
    ax0.set_xticks([])
    ymin, ymax = ax0.get_ylim()
    xmin, xmax = ax0.get_xlim()
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    # ax0.spines['left'].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.set_xlabel("")

    ax1 = fig.add_subplot(gs[0:3, 3])
    ax1.hist(all_y, bins=100, orientation="horizontal", alpha=0.5)
    ax1.hist(real_y, bins=100, orientation="horizontal", color="k", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_yticks([])
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel("# spikes")

    ax2 = fig.add_subplot(gs[3, 0:3])
    ax2.hist(all_x, bins=100, alpha=0.5)
    ax2.hist(real_x, bins=100, color="k", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel(r"x ($\mu$m)")
    ax2.set_ylabel("# spikes")


def plot_comparison_precision(benchmarks):
    import pylab as plt

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 10), squeeze=False)

    for bench in benchmarks:
        # gt_positions = bench.gt_positions
        # template_positions = bench.template_positions
        # dx = np.abs(gt_positions[:, 0] - template_positions[:, 0])
        # dy = np.abs(gt_positions[:, 1] - template_positions[:, 1])
        # dz = np.abs(gt_positions[:, 2] - template_positions[:, 2])
        # ax = axes[0, 0]
        # ax.errorbar(np.arange(3), [dx.mean(), dy.mean(), dz.mean()], yerr=[dx.std(), dy.std(), dz.std()], label=bench.title)

        spikes = bench.spike_positions[0]
        units = bench.waveforms.sorting.unit_ids
        all_x = np.concatenate([spikes[unit_id]["x"] for unit_id in units])
        all_y = np.concatenate([spikes[unit_id]["y"] for unit_id in units])
        all_z = np.concatenate([spikes[unit_id]["z"] for unit_id in units])

        gt_positions = bench.gt_positions
        real_x = np.concatenate([gt_positions[c, 0] * np.ones(len(spikes[i]["x"])) for c, i in enumerate(units)])
        real_y = np.concatenate([gt_positions[c, 1] * np.ones(len(spikes[i]["y"])) for c, i in enumerate(units)])
        real_z = np.concatenate([gt_positions[c, 2] * np.ones(len(spikes[i]["z"])) for c, i in enumerate(units)])

        dx = np.abs(all_x - real_x)
        dy = np.abs(all_y - real_y)
        dz = np.abs(all_z - real_z)
        ax = axes[0, 0]
        ax.errorbar(
            np.arange(3), [dx.mean(), dy.mean(), dz.mean()], yerr=[dx.std(), dy.std(), dz.std()], label=bench.title
        )
    ax.legend()
    ax.set_ylabel("error (um)")
    ax.set_xticks(np.arange(3), ["x", "y", "z"])
    _simpleaxis(ax)

    x_means = []
    x_stds = []
    for count, bench in enumerate(benchmarks):
        x_means += [np.mean(bench.means_over_templates)]
        x_stds += [np.std(bench.means_over_templates)]

    # ax.set_yticks([])
    # ax.set_ylim(ymin, ymax)

    ax = axes[0, 1]
    _simpleaxis(ax)

    y_means = []
    y_stds = []
    for count, bench in enumerate(benchmarks):
        y_means += [np.mean(bench.stds_over_templates)]
        y_stds += [np.std(bench.stds_over_templates)]

    colors = [f"C{i}" for i in range(len(x_means))]
    ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt=".", c="0.5", alpha=0.5)
    ax.scatter(x_means, y_means, c=colors, s=200)

    ax.set_ylabel("error variances (um)")
    ax.set_xlabel("error means (um)")
    # ax.set_yticks([]
    ymin, ymax = ax.get_ylim()
    # ax.set_ylim(0, 25)
    ax.legend()


def plot_figure_1(benchmark, mode="average", cell_ind="auto"):
    if cell_ind == "auto":
        norms = np.linalg.norm(benchmark.gt_positions[:, :2], axis=1)
        cell_ind = np.argsort(norms)[0]

    import pylab as plt

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    plot_probe_map(benchmark.recording, ax=axs[0, 0])
    axs[0, 0].scatter(benchmark.gt_positions[:, 0], benchmark.gt_positions[:, 1], c="k")
    axs[0, 0].scatter(benchmark.gt_positions[cell_ind, 0], benchmark.gt_positions[cell_ind, 1], c="r")
    plt.rc("font", size=13)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)

    import spikeinterface.full as si

    sorting = benchmark.waveforms.sorting
    unit_id = sorting.unit_ids[cell_ind]

    spikes_seg0 = sorting.to_spike_vector(concatenated=False)[0]
    mask = spikes_seg0["unit_index"] == cell_ind
    times = spikes_seg0[mask] / sorting.get_sampling_frequency()

    print(benchmark.recording)
    # si.plot_traces(benchmark.recording, mode='line', time_range=(times[0]-0.01, times[0] + 0.1), channel_ids=benchmark.recording.channel_ids[:20], ax=axs[0, 1])
    # axs[0, 1].set_ylabel('Neurons')

    # si.plot_spikes_on_traces(benchmark.waveforms, unit_ids=[unit_id], time_range=(times[0]-0.01, times[0] + 0.1), unit_colors={unit_id : 'r'}, ax=axs[0, 1],
    #    channel_ids=benchmark.recording.channel_ids[120:180], )

    waveforms = extract_waveforms(
        benchmark.recording,
        benchmark.gt_sorting,
        None,
        mode="memory",
        ms_before=2.5,
        ms_after=2.5,
        max_spikes_per_unit=100,
        return_scaled=False,
        **benchmark.job_kwargs,
        sparse=True,
        method="radius",
        radius_um=100,
    )

    unit_id = waveforms.sorting.unit_ids[cell_ind]

    si.plot_unit_templates(waveforms, unit_ids=[unit_id], ax=axs[1, 0], same_axis=True, unit_colors={unit_id: "r"})
    ymin, ymax = axs[1, 0].get_ylim()
    xmin, xmax = axs[1, 0].get_xlim()
    axs[1, 0].set_title("Averaged template")
    si.plot_unit_waveforms(waveforms, unit_ids=[unit_id], ax=axs[1, 1], same_axis=True, unit_colors={unit_id: "r"})
    axs[1, 1].set_xlim(xmin, xmax)
    axs[1, 1].set_ylim(ymin, ymax)
    axs[1, 1].set_title("Single spikes")

    for i in [0, 1]:
        for j in [0, 1]:
            axs[i, j].spines["top"].set_visible(False)
            axs[i, j].spines["right"].set_visible(False)

    for i in [1]:
        for j in [0, 1]:
            axs[i, j].spines["left"].set_visible(False)
            axs[i, j].spines["bottom"].set_visible(False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_title("")
