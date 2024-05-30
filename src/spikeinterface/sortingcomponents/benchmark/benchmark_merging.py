from __future__ import annotations

from spikeinterface.sortingcomponents.merging import merge_spikes
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import CollisionGTComparison, compare_sorter_to_ground_truth
from spikeinterface.widgets import (
    plot_agreement_matrix,
    plot_comparison_collision_by_similarity,
)

import pylab as plt
import matplotlib.patches as mpatches
import numpy as np
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import Benchmark, BenchmarkStudy
from spikeinterface.core.basesorting import minimum_spike_dtype


class MergingBenchmark(Benchmark):

    def __init__(self, recording, splitted_sorting, params, gt_sorting):
        self.recording = recording
        self.splitted_sorting = splitted_sorting
        self.method = params["method"]
        self.gt_sorting = gt_sorting
        self.method_kwargs = params["method_kwargs"]
        self.result = {}

    def run(self, **job_kwargs):
        self.result['sorting'] = merge_spikes(
            self.recording, self.splitted_sorting, method=self.method, method_kwargs=self.method_kwargs
        )

    def compute_result(self, **result_params):
        sorting = self.result["sorting"]
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=True)
        self.result["gt_comparison"] = comp
        
    _run_key_saved = [
        ("sorting", "sorting"),
    ]
    _result_key_saved = [("gt_comparison", "pickle")]


class MergingStudy(BenchmarkStudy):

    benchmark_class = MergingBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        benchmark = MergingBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark

    def plot_agreements(self, case_keys=None, figsize=(15, 15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            ax.set_title(self.cases[key]["label"])
            plot_agreement_matrix(self.get_result(key)["gt_comparison"], ax=ax)

        return fig