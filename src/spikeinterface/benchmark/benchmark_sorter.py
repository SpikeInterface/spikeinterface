"""
This replace the previous `GroundTruthStudy`
"""

import numpy as np
from spikeinterface.core import NumpySorting
from .benchmark_base import Benchmark, BenchmarkStudy, MixinStudyUnitCount
from spikeinterface.sorters import run_sorter
from spikeinterface.comparison import compare_sorter_to_ground_truth


# TODO later integrate CollisionGTComparison optionally in this class.


class SorterBenchmark(Benchmark):
    def __init__(self, recording, gt_sorting, params, sorter_folder):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.params = params
        self.sorter_folder = sorter_folder
        self.result = {}

    def run(self):
        # run one sorter sorter_name is must be in params
        raw_sorting = run_sorter(recording=self.recording, folder=self.sorter_folder, **self.params)
        sorting = NumpySorting.from_sorting(raw_sorting)
        self.result = {"sorting": sorting}

    def compute_result(self, exhaustive_gt=True):
        # run becnhmark result
        sorting = self.result["sorting"]
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=exhaustive_gt)
        self.result["gt_comparison"] = comp

    _run_key_saved = [
        ("sorting", "sorting"),
    ]
    _result_key_saved = [
        ("gt_comparison", "pickle"),
    ]


class SorterStudy(BenchmarkStudy, MixinStudyUnitCount):
    """
    This class is used to tests several sorter in several situtation.
    This replace the previous GroundTruthStudy with more flexibility.
    """

    benchmark_class = SorterBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        sorter_folder = self.folder / "sorters" / self.key_to_str(key)
        benchmark = SorterBenchmark(recording, gt_sorting, params, sorter_folder)
        return benchmark

    def remove_benchmark(self, key):
        BenchmarkStudy.remove_benchmark(self, key)

        sorter_folder = self.folder / "sorters" / self.key_to_str(key)
        import shutil

        if sorter_folder.exists():
            shutil.rmtree(sorter_folder)

    # plotting as methods
    def plot_unit_counts(self, **kwargs):
        from .benchmark_plot_tools import plot_unit_counts

        return plot_unit_counts(self, **kwargs)

    def plot_performances(self, **kwargs):
        from .benchmark_plot_tools import plot_performances

        return plot_performances(self, **kwargs)

    def plot_performances_vs_snr(self, **kwargs):
        from .benchmark_plot_tools import plot_performances_vs_snr

        return plot_performances_vs_snr(self, **kwargs)

    def plot_performances_ordered(self, **kwargs):
        from .benchmark_plot_tools import plot_performances_ordered

        return plot_performances_ordered(self, **kwargs)

    def plot_performances_swarm(self, **kwargs):
        from .benchmark_plot_tools import plot_performances_swarm

        return plot_performances_swarm(self, **kwargs)

    def plot_agreement_matrix(self, **kwargs):
        from .benchmark_plot_tools import plot_agreement_matrix

        return plot_agreement_matrix(self, **kwargs)

    def plot_performance_losses(self, *args, **kwargs):
        from .benchmark_plot_tools import plot_performance_losses

        return plot_performance_losses(self, *args, **kwargs)
