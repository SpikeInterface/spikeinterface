from __future__ import annotations

from spikeinterface.postprocessing.localization_tools import (
    compute_center_of_mass,
    compute_monopolar_triangulation,
    compute_grid_convolution,
)
import numpy as np
from .benchmark_base import Benchmark, BenchmarkStudy
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer


class PeakLocalizationBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, gt_positions, channel_from_template=False):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.gt_positions = gt_positions
        self.channel_from_template = channel_from_template
        self.params = params
        self.result = {}
        self.templates_params = {}
        for key in ["ms_before", "ms_after"]:
            self.params[key] = self.params.get(key, 2)
            self.templates_params[key] = self.params[key]

        if not self.channel_from_template:
            self.params["spike_retriver_kwargs"] = {"channel_from_template": False}
        else:
            ## TODO
            pass

    def run(self, **job_kwargs):
        sorting_analyzer = create_sorting_analyzer(
            self.gt_sorting, self.recording, format="memory", sparse=False, **job_kwargs
        )
        sorting_analyzer.compute("random_spikes")
        ext = sorting_analyzer.compute("templates", **self.templates_params, **job_kwargs)
        templates = ext.get_data(outputs="Templates")
        ext = sorting_analyzer.compute("spike_locations", **self.params, **job_kwargs)
        spikes_locations = ext.get_data(outputs="by_unit")
        self.result = {"spikes_locations": spikes_locations}
        self.result["templates"] = templates

    def compute_result(self, **result_params):
        errors = {}

        for unit_ind, unit_id in enumerate(self.gt_sorting.unit_ids):
            data = self.result["spikes_locations"][0][unit_id]
            errors[unit_id] = np.sqrt(
                (data["x"] - self.gt_positions[unit_ind, 0]) ** 2 + (data["y"] - self.gt_positions[unit_ind, 1]) ** 2
            )

        self.result["medians_over_templates"] = np.array(
            [np.median(errors[unit_id]) for unit_id in self.gt_sorting.unit_ids]
        )
        self.result["mads_over_templates"] = np.array(
            [np.median(np.abs(errors[unit_id] - np.median(errors[unit_id]))) for unit_id in self.gt_sorting.unit_ids]
        )
        self.result["errors"] = errors

    _run_key_saved = [
        ("spikes_locations", "pickle"),
        ("templates", "zarr_templates"),
    ]
    _result_key_saved = [
        ("errors", "pickle"),
        ("medians_over_templates", "npy"),
        ("mads_over_templates", "npy"),
    ]


class PeakLocalizationStudy(BenchmarkStudy):

    benchmark_class = PeakLocalizationBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        benchmark = PeakLocalizationBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark

    def plot_comparison_positions(self, case_keys=None):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))

        for count, key in enumerate(case_keys):
            analyzer = self.get_sorting_analyzer(key)
            metrics = analyzer.get_extension("quality_metrics").get_data()
            snrs = metrics["snr"].values
            result = self.get_result(key)
            norms = np.linalg.norm(result["templates"].templates_array, axis=(1, 2))

            coordinates = self.benchmarks[key].gt_positions[:, :2].copy()
            coordinates[:, 0] -= coordinates[:, 0].mean()
            coordinates[:, 1] -= coordinates[:, 1].mean()
            distances_to_center = np.linalg.norm(coordinates, axis=1)
            zdx = np.argsort(distances_to_center)
            idx = np.argsort(norms)

            wdx = np.argsort(snrs)

            data = result["medians_over_templates"]

            axs[0].plot(snrs[wdx], data[wdx], lw=2, label=self.cases[key]["label"])
            ymin = (data - result["mads_over_templates"])[wdx]
            ymax = (data + result["mads_over_templates"])[wdx]

            axs[0].fill_between(snrs[wdx], ymin, ymax, alpha=0.5)
            axs[0].set_xlabel("snr")
            axs[0].set_ylabel("error (um)")

            axs[1].plot(
                distances_to_center[zdx],
                data[zdx],
                lw=2,
                label=self.cases[key]["label"],
            )
            ymin = (data - result["mads_over_templates"])[zdx]
            ymax = (data + result["mads_over_templates"])[zdx]

            axs[1].fill_between(distances_to_center[zdx], ymin, ymax, alpha=0.5)
            axs[1].set_xlabel("distance to center (um)")

        x_means = []
        x_stds = []
        for count, key in enumerate(case_keys):
            result = self.get_result(key)["medians_over_templates"]
            x_means += [result.mean()]
            x_stds += [result.std()]

        y_means = []
        y_stds = []
        for count, key in enumerate(case_keys):
            result = self.get_result(key)["mads_over_templates"]
            y_means += [result.mean()]
            y_stds += [result.std()]

        colors = [f"C{i}" for i in range(len(x_means))]
        axs[2].errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt=".", c="0.5", alpha=0.5)
        axs[2].scatter(x_means, y_means, c=colors, s=200)

        axs[2].set_ylabel("error mads (um)")
        axs[2].set_xlabel("error medians (um)")
        ymin, ymax = axs[2].get_ylim()
        axs[2].set_ylim(0, 12)
        axs[1].legend()


class UnitLocalizationBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, gt_positions):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.gt_positions = gt_positions
        assert "method" in params, "Method should be specified in the params!"
        self.method = params["method"]
        self.params = params["method_kwargs"]
        self.result = {}
        self.waveforms_params = {}
        for key in ["ms_before", "ms_after"]:
            self.waveforms_params[key] = self.params.pop(key, 2)

    def run(self, **job_kwargs):
        sorting_analyzer = create_sorting_analyzer(self.gt_sorting, self.recording, format="memory", **job_kwargs)
        sorting_analyzer.compute("random_spikes")
        sorting_analyzer.compute("waveforms", **self.waveforms_params, **job_kwargs)
        ext = sorting_analyzer.compute("templates", **job_kwargs)
        templates = ext.get_data(outputs="Templates")

        if self.method == "center_of_mass":
            unit_locations = compute_center_of_mass(sorting_analyzer, **self.params)
        elif self.method == "monopolar_triangulation":
            unit_locations = compute_monopolar_triangulation(sorting_analyzer, **self.params)
        elif self.method == "grid_convolution":
            unit_locations = compute_grid_convolution(sorting_analyzer, **self.params)

        if unit_locations.shape[1] == 2:
            unit_locations = np.hstack((unit_locations, np.zeros((len(unit_locations), 1))))

        self.result = {"unit_locations": unit_locations}
        self.result["templates"] = templates

    def compute_result(self, **result_params):
        errors = np.linalg.norm(self.gt_positions[:, :2] - self.result["unit_locations"][:, :2], axis=1)
        self.result["errors"] = errors

    _run_key_saved = [
        ("unit_locations", "npy"),
        ("templates", "zarr_templates"),
    ]
    _result_key_saved = [("errors", "npy")]


class UnitLocalizationStudy(BenchmarkStudy):

    benchmark_class = UnitLocalizationBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        init_kwargs = self.cases[key]["init_kwargs"]
        params = self.cases[key]["params"]
        benchmark = UnitLocalizationBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark

    def plot_template_errors(self, case_keys=None, show_probe=True):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))

        from spikeinterface.widgets import plot_probe_map

        for count, key in enumerate(case_keys):
            gt_positions = self.benchmarks[key].gt_positions
            colors = np.arange(len(gt_positions))
            if show_probe:
                plot_probe_map(self.benchmarks[key].recording, ax=axs[count])
            axs[count].scatter(gt_positions[:, 0], gt_positions[:, 1], c=colors)
            axs[count].set_title(self.cases[key]["label"])

            result = self.get_result(key)
            axs[count].scatter(
                result["unit_locations"][:, 0],
                result["unit_locations"][:, 1],
                c=colors,
                marker="v",
            )

    def plot_comparison_positions(self, case_keys=None):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import pylab as plt

        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))

        for count, key in enumerate(case_keys):
            analyzer = self.get_sorting_analyzer(key)
            metrics = analyzer.get_extension("quality_metrics").get_data()
            snrs = metrics["snr"].values
            result = self.get_result(key)
            norms = np.linalg.norm(result["templates"].templates_array, axis=(1, 2))

            coordinates = self.benchmarks[key].gt_positions[:, :2].copy()
            coordinates[:, 0] -= coordinates[:, 0].mean()
            coordinates[:, 1] -= coordinates[:, 1].mean()
            distances_to_center = np.linalg.norm(coordinates, axis=1)
            zdx = np.argsort(distances_to_center)
            idx = np.argsort(norms)

            from scipy.signal import savgol_filter

            wdx = np.argsort(snrs)

            data = result["errors"]

            axs[0].plot(snrs[wdx], data[wdx], lw=2, label=self.cases[key]["label"])

            axs[0].set_xlabel("snr")
            axs[0].set_ylabel("error (um)")

            axs[1].plot(
                distances_to_center[zdx],
                data[zdx],
                lw=2,
                label=self.cases[key]["label"],
            )

            axs[1].legend()
            axs[1].set_xlabel("distance to center (um)")
            axs[2].bar([count], np.mean(data), yerr=np.std(data))


# def plot_comparison_inferences(benchmarks, bin_size=np.arange(0.1, 20, 1)):
#     import numpy as np
#     import sklearn
#     import scipy.stats
#     import spikeinterface.full as si

#     plt.rc("font", size=11)
#     plt.rc("xtick", labelsize=12)
#     plt.rc("ytick", labelsize=12)

#     from scipy.signal import savgol_filter

#     smoothing_factor = 5

#     fig = plt.figure(figsize=(10, 12))
#     gs = fig.add_gridspec(8, 10)

#     ax3 = fig.add_subplot(gs[0:2, 6:10])
#     ax4 = fig.add_subplot(gs[2:4, 6:10])

#     ax3.spines["top"].set_visible(False)
#     ax3.spines["right"].set_visible(False)
#     ax3.set_ylabel("correlation coefficient")
#     ax3.set_xticks([])

#     ax4.spines["top"].set_visible(False)
#     ax4.spines["right"].set_visible(False)
#     ax4.set_ylabel("chi squared")
#     ax4.set_xlabel("bin size (um)")

#     def chiSquared(p, q):
#         return 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-6))

#     for count, benchmark in enumerate(benchmarks):
#         spikes = benchmark.spike_positions[0]
#         units = benchmark.waveforms.sorting.unit_ids
#         all_x = np.concatenate([spikes[unit_id]["x"] for unit_id in units])
#         all_y = np.concatenate([spikes[unit_id]["y"] for unit_id in units])

#         gt_positions = benchmark.gt_positions[:, :2]
#         real_x = np.concatenate([gt_positions[c, 0] * np.ones(len(spikes[i]["x"])) for c, i in enumerate(units)])
#         real_y = np.concatenate([gt_positions[c, 1] * np.ones(len(spikes[i]["x"])) for c, i in enumerate(units)])

#         r_y = np.zeros(len(bin_size))
#         c_y = np.zeros(len(bin_size))
#         for i, b in enumerate(bin_size):
#             all_bins = np.arange(all_y.min(), all_y.max(), b)
#             x1, y2 = np.histogram(all_y, bins=all_bins)
#             x2, y2 = np.histogram(real_y, bins=all_bins)

#             r_y[i] = np.corrcoef(x1, x2)[0, 1]
#             c_y[i] = chiSquared(x1, x2)

#         r_x = np.zeros(len(bin_size))
#         c_x = np.zeros(len(bin_size))
#         for i, b in enumerate(bin_size):
#             all_bins = np.arange(all_x.min(), all_x.max(), b)
#             x1, y2 = np.histogram(all_x, bins=all_bins)
#             x2, y2 = np.histogram(real_x, bins=all_bins)

#             r_x[i] = np.corrcoef(x1, x2)[0, 1]
#             c_x[i] = chiSquared(x1, x2)

#         ax3.plot(bin_size, savgol_filter((r_y + r_x) / 2, smoothing_factor, 3), c=f"C{count}", label=benchmark.title)
#         ax4.plot(bin_size, savgol_filter((c_y + c_x) / 2, smoothing_factor, 3), c=f"C{count}", label=benchmark.title)

#     r_control_y = np.zeros(len(bin_size))
#     c_control_y = np.zeros(len(bin_size))
#     for i, b in enumerate(bin_size):
#         all_bins = np.arange(all_y.min(), all_y.max(), b)
#         random_y = all_y.min() + (all_y.max() - all_y.min()) * np.random.rand(len(all_y))
#         x1, y2 = np.histogram(random_y, bins=all_bins)
#         x2, y2 = np.histogram(real_y, bins=all_bins)

#         r_control_y[i] = np.corrcoef(x1, x2)[0, 1]
#         c_control_y[i] = chiSquared(x1, x2)

#     r_control_x = np.zeros(len(bin_size))
#     c_control_x = np.zeros(len(bin_size))
#     for i, b in enumerate(bin_size):
#         all_bins = np.arange(all_x.min(), all_x.max(), b)
#         random_x = all_x.min() + (all_x.max() - all_x.min()) * np.random.rand(len(all_y))
#         x1, y2 = np.histogram(random_x, bins=all_bins)
#         x2, y2 = np.histogram(real_x, bins=all_bins)

#         r_control_x[i] = np.corrcoef(x1, x2)[0, 1]
#         c_control_x[i] = chiSquared(x1, x2)

#     ax3.plot(bin_size, savgol_filter((r_control_y + r_control_x) / 2, smoothing_factor, 3), "0.5", label="Control")
#     ax4.plot(bin_size, savgol_filter((c_control_y + c_control_x) / 2, smoothing_factor, 3), "0.5", label="Control")

#     ax4.legend()

#     ax0 = fig.add_subplot(gs[0:3, 0:3])

#     si.plot_probe_map(benchmarks[0].recording, ax=ax0)
#     ax0.scatter(all_x, all_y, alpha=0.5)
#     ax0.scatter(gt_positions[:, 0], gt_positions[:, 1], c="k")
#     ax0.set_xticks([])
#     ymin, ymax = ax0.get_ylim()
#     xmin, xmax = ax0.get_xlim()
#     ax0.spines["top"].set_visible(False)
#     ax0.spines["right"].set_visible(False)
#     # ax0.spines['left'].set_visible(False)
#     ax0.spines["bottom"].set_visible(False)
#     ax0.set_xlabel("")

#     ax1 = fig.add_subplot(gs[0:3, 3])
#     ax1.hist(all_y, bins=100, orientation="horizontal", alpha=0.5)
#     ax1.hist(real_y, bins=100, orientation="horizontal", color="k", alpha=0.5)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.set_yticks([])
#     ax1.set_ylim(ymin, ymax)
#     ax1.set_xlabel("# spikes")

#     ax2 = fig.add_subplot(gs[3, 0:3])
#     ax2.hist(all_x, bins=100, alpha=0.5)
#     ax2.hist(real_x, bins=100, color="k", alpha=0.5)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.set_xlim(xmin, xmax)
#     ax2.set_xlabel(r"x ($\\mu$m)")
#     ax2.set_ylabel("# spikes")


# def plot_comparison_precision(benchmarks):
#     import pylab as plt

#     fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 10), squeeze=False)

#     for bench in benchmarks:
#         # gt_positions = bench.gt_positions
#         # template_positions = bench.template_positions
#         # dx = np.abs(gt_positions[:, 0] - template_positions[:, 0])
#         # dy = np.abs(gt_positions[:, 1] - template_positions[:, 1])
#         # dz = np.abs(gt_positions[:, 2] - template_positions[:, 2])
#         # ax = axes[0, 0]
#         # ax.errorbar(np.arange(3), [dx.mean(), dy.mean(), dz.mean()], yerr=[dx.std(), dy.std(), dz.std()], label=bench.title)

#         spikes = bench.spike_positions[0]
#         units = bench.waveforms.sorting.unit_ids
#         all_x = np.concatenate([spikes[unit_id]["x"] for unit_id in units])
#         all_y = np.concatenate([spikes[unit_id]["y"] for unit_id in units])
#         all_z = np.concatenate([spikes[unit_id]["z"] for unit_id in units])

#         gt_positions = bench.gt_positions
#         real_x = np.concatenate([gt_positions[c, 0] * np.ones(len(spikes[i]["x"])) for c, i in enumerate(units)])
#         real_y = np.concatenate([gt_positions[c, 1] * np.ones(len(spikes[i]["y"])) for c, i in enumerate(units)])
#         real_z = np.concatenate([gt_positions[c, 2] * np.ones(len(spikes[i]["z"])) for c, i in enumerate(units)])

#         dx = np.abs(all_x - real_x)
#         dy = np.abs(all_y - real_y)
#         dz = np.abs(all_z - real_z)
#         ax = axes[0, 0]
#         ax.errorbar(
#             np.arange(3), [dx.mean(), dy.mean(), dz.mean()], yerr=[dx.std(), dy.std(), dz.std()], label=bench.title
#         )
#     ax.legend()
#     ax.set_ylabel("error (um)")
#     ax.set_xticks(np.arange(3), ["x", "y", "z"])
#     despine(ax)

#     x_means = []
#     x_stds = []
#     for count, bench in enumerate(benchmarks):
#         x_means += [np.mean(bench.means_over_templates)]
#         x_stds += [np.std(bench.means_over_templates)]

#     # ax.set_yticks([])
#     # ax.set_ylim(ymin, ymax)

#     ax = axes[0, 1]
#     despine(ax)

#     y_means = []
#     y_stds = []
#     for count, bench in enumerate(benchmarks):
#         y_means += [np.mean(bench.stds_over_templates)]
#         y_stds += [np.std(bench.stds_over_templates)]

#     colors = [f"C{i}" for i in range(len(x_means))]
#     ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt=".", c="0.5", alpha=0.5)
#     ax.scatter(x_means, y_means, c=colors, s=200)

#     ax.set_ylabel("error variances (um)")
#     ax.set_xlabel("error means (um)")
#     # ax.set_yticks([]
#     ymin, ymax = ax.get_ylim()
#     # ax.set_ylim(0, 25)
#     ax.legend()


# def plot_figure_1(benchmark, mode="average", cell_ind="auto"):
#     if cell_ind == "auto":
#         norms = np.linalg.norm(benchmark.gt_positions[:, :2], axis=1)
#         cell_ind = np.argsort(norms)[0]

#     import pylab as plt

#     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
#     plot_probe_map(benchmark.recording, ax=axs[0, 0])
#     axs[0, 0].scatter(benchmark.gt_positions[:, 0], benchmark.gt_positions[:, 1], c="k")
#     axs[0, 0].scatter(benchmark.gt_positions[cell_ind, 0], benchmark.gt_positions[cell_ind, 1], c="r")
#     plt.rc("font", size=13)
#     plt.rc("xtick", labelsize=12)
#     plt.rc("ytick", labelsize=12)

#     import spikeinterface.full as si

#     sorting = benchmark.waveforms.sorting
#     unit_id = sorting.unit_ids[cell_ind]

#     spikes_seg0 = sorting.to_spike_vector(concatenated=False)[0]
#     mask = spikes_seg0["unit_index"] == cell_ind
#     times = spikes_seg0[mask] / sorting.get_sampling_frequency()

#     print(benchmark.recording)
#     # si.plot_traces(benchmark.recording, mode='line', time_range=(times[0]-0.01, times[0] + 0.1), channel_ids=benchmark.recording.channel_ids[:20], ax=axs[0, 1])
#     # axs[0, 1].set_ylabel('Neurons')

#     # si.plot_spikes_on_traces(benchmark.waveforms, unit_ids=[unit_id], time_range=(times[0]-0.01, times[0] + 0.1), unit_colors={unit_id : 'r'}, ax=axs[0, 1],
#     #    channel_ids=benchmark.recording.channel_ids[120:180], )

#     waveforms = extract_waveforms(
#         benchmark.recording,
#         benchmark.gt_sorting,
#         None,
#         mode="memory",
#         ms_before=2.5,
#         ms_after=2.5,
#         max_spikes_per_unit=100,
#         return_scaled=False,
#         **benchmark.job_kwargs,
#         sparse=True,
#         method="radius",
#         radius_um=100,
#     )

#     unit_id = waveforms.sorting.unit_ids[cell_ind]

#     si.plot_unit_templates(waveforms, unit_ids=[unit_id], ax=axs[1, 0], same_axis=True, unit_colors={unit_id: "r"})
#     ymin, ymax = axs[1, 0].get_ylim()
#     xmin, xmax = axs[1, 0].get_xlim()
#     axs[1, 0].set_title("Averaged template")
#     si.plot_unit_waveforms(waveforms, unit_ids=[unit_id], ax=axs[1, 1], same_axis=True, unit_colors={unit_id: "r"})
#     axs[1, 1].set_xlim(xmin, xmax)
#     axs[1, 1].set_ylim(ymin, ymax)
#     axs[1, 1].set_title("Single spikes")

#     for i in [0, 1]:
#         for j in [0, 1]:
#             axs[i, j].spines["top"].set_visible(False)
#             axs[i, j].spines["right"].set_visible(False)

#     for i in [1]:
#         for j in [0, 1]:
#             axs[i, j].spines["left"].set_visible(False)
#             axs[i, j].spines["bottom"].set_visible(False)
#             axs[i, j].set_xticks([])
#             axs[i, j].set_yticks([])
#             axs[i, j].set_title("")
