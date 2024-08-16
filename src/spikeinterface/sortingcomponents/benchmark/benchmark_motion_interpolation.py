from __future__ import annotations

import numpy as np


from spikeinterface.sorters import run_sorter

from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.sortingcomponents.motion.motion_interpolation import InterpolateMotionRecording
from spikeinterface.curation import MergeUnitsSorting


from spikeinterface.sortingcomponents.benchmark.benchmark_tools import Benchmark, BenchmarkStudy, _simpleaxis


class MotionInterpolationBenchmark(Benchmark):
    def __init__(
        self,
        static_recording,
        gt_sorting,
        params,
        sorter_folder,
        drifting_recording,
        motion,
        temporal_bins,
        spatial_bins,
    ):
        Benchmark.__init__(self)
        self.static_recording = static_recording
        self.gt_sorting = gt_sorting
        self.params = params

        self.sorter_folder = sorter_folder
        self.drifting_recording = drifting_recording
        self.motion = motion
        self.temporal_bins = temporal_bins
        self.spatial_bins = spatial_bins

    def run(self, **job_kwargs):

        if self.params["recording_source"] == "static":
            recording = self.static_recording
        elif self.params["recording_source"] == "drifting":
            recording = self.drifting_recording
        elif self.params["recording_source"] == "corrected":
            correct_motion_kwargs = self.params["correct_motion_kwargs"]
            recording = InterpolateMotionRecording(self.drifting_recording, self.motion, **correct_motion_kwargs)
        else:
            raise ValueError("recording_source")

        sorter_name = self.params["sorter_name"]
        sorter_params = self.params["sorter_params"]
        sorting = run_sorter(
            sorter_name,
            recording,
            output_folder=self.sorter_folder,
            **sorter_params,
            delete_output_folder=False,
        )

        self.result["sorting"] = sorting

    def compute_result(self, exhaustive_gt=True, merging_score=0.2):
        sorting = self.result["sorting"]
        # self.result[""] =
        comparison = GroundTruthComparison(self.gt_sorting, sorting, exhaustive_gt=exhaustive_gt)
        self.result["comparison"] = comparison
        self.result["accuracy"] = comparison.get_performance()["accuracy"].values.astype("float32")

        gt_unit_ids = self.gt_sorting.unit_ids
        unit_ids = sorting.unit_ids

        # find best merges
        scores = comparison.agreement_scores
        to_merge = []
        for gt_unit_id in gt_unit_ids:
            (inds,) = np.nonzero(scores.loc[gt_unit_id, :].values > merging_score)
            merge_ids = unit_ids[inds]
            if merge_ids.size > 1:
                to_merge.append(list(merge_ids))

        merged_sporting = MergeUnitsSorting(sorting, to_merge)
        comparison_merged = GroundTruthComparison(self.gt_sorting, merged_sporting, exhaustive_gt=True)

        self.result["comparison_merged"] = comparison_merged
        self.result["accuracy_merged"] = comparison_merged.get_performance()["accuracy"].values.astype("float32")

    _run_key_saved = [
        ("sorting", "sorting"),
    ]
    _result_key_saved = [
        ("comparison", "pickle"),
        ("accuracy", "npy"),
        ("comparison_merged", "pickle"),
        ("accuracy_merged", "npy"),
    ]


class MotionInterpolationStudy(BenchmarkStudy):

    benchmark_class = MotionInterpolationBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        sorter_folder = self.folder / "sorters" / self.key_to_str(key)
        sorter_folder.parent.mkdir(exist_ok=True)
        benchmark = MotionInterpolationBenchmark(
            recording, gt_sorting, params, sorter_folder=sorter_folder, **init_kwargs
        )
        return benchmark

    def plot_sorting_accuracy(
        self,
        case_keys=None,
        mode="ordered_accuracy",
        legend=True,
        colors=None,
        mode_best_merge=False,
        figsize=(10, 5),
        ax=None,
        axes=None,
    ):
        import matplotlib.pyplot as plt

        if case_keys is None:
            case_keys = list(self.cases.keys())

        if not mode_best_merge:
            ls = "-"
        else:
            ls = "--"

        if mode == "ordered_accuracy":
            if ax is None:

                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure

            order = None
            for i, key in enumerate(case_keys):
                result = self.get_result(key)
                if not mode_best_merge:
                    accuracy = result["accuracy"]
                else:
                    accuracy = result["accuracy_merged"]
                label = self.cases[key]["label"]
                color = colors[i] if colors is not None else None
                order = np.argsort(accuracy)[::-1]
                accuracy = accuracy[order]
                ax.plot(accuracy, label=label, ls=ls, color=color)
            if legend:
                ax.legend()
            ax.set_ylabel("accuracy")
            ax.set_xlabel("units ordered by accuracy")

        elif mode == "depth_snr":
            if axes is None:
                fig, axs = plt.subplots(nrows=len(case_keys), figsize=figsize, sharey=True, sharex=True)
            else:
                fig = axes[0].figure
                axs = axes

            for i, key in enumerate(case_keys):
                ax = axs[i]
                result = self.get_result(key)
                if not mode_best_merge:
                    accuracy = result["accuracy"]
                else:
                    accuracy = result["accuracy_merged"]
                label = self.cases[key]["label"]

                analyzer = self.get_sorting_analyzer(key)
                ext = analyzer.get_extension("unit_locations")
                if ext is None:
                    ext = analyzer.compute("unit_locations")
                unit_locations = ext.get_data()
                unit_depth = unit_locations[:, 1]

                snr = analyzer.get_extension("quality_metrics").get_data()["snr"].values

                points = ax.scatter(unit_depth, snr, c=accuracy)
                points.set_clim(0.0, 1.0)
                ax.set_title(label)

                chan_locations = analyzer.get_channel_locations()

                ax.axvline(np.min(chan_locations[:, 1]), ls="--", color="k")
                ax.axvline(np.max(chan_locations[:, 1]), ls="--", color="k")
                ax.set_ylabel("snr")
            ax.set_xlabel("depth")

            cbar = fig.colorbar(points, ax=axs[:], location="right", shrink=0.6)
            cbar.ax.set_ylabel("accuracy")

        elif mode == "snr":
            fig, ax = plt.subplots(figsize=figsize)

            for i, key in enumerate(case_keys):
                result = self.get_result(key)
                label = self.cases[key]["label"]
                if not mode_best_merge:
                    accuracy = result["accuracy"]
                else:
                    accuracy = result["accuracy_merged"]

                analyzer = self.get_sorting_analyzer(key)
                snr = analyzer.get_extension("quality_metrics").get_data()["snr"].values

                ax.scatter(snr, accuracy, label=label)
            ax.set_xlabel("snr")
            ax.set_ylabel("accuracy")

            ax.legend()

        elif mode == "depth":
            fig, ax = plt.subplots(figsize=figsize)

            for i, key in enumerate(case_keys):
                result = self.get_result(key)
                label = self.cases[key]["label"]
                if not mode_best_merge:
                    accuracy = result["accuracy"]
                else:
                    accuracy = result["accuracy_merged"]
                analyzer = self.get_sorting_analyzer(key)

                ext = analyzer.get_extension("unit_locations")
                if ext is None:
                    ext = analyzer.compute("unit_locations")
                unit_locations = ext.get_data()
                unit_depth = unit_locations[:, 1]

                ax.scatter(unit_depth, accuracy, label=label)

            chan_locations = analyzer.get_channel_locations()

            ax.axvline(np.min(chan_locations[:, 1]), ls="--", color="k")
            ax.axvline(np.max(chan_locations[:, 1]), ls="--", color="k")
            ax.legend()
            ax.set_xlabel("depth")
            ax.set_ylabel("accuracy")

        return fig
