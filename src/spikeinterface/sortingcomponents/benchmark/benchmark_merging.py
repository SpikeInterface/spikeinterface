from __future__ import annotations

from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
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

    def __init__(self, recording, splitted_sorting, params):
        self.recording = recording
        self.splitted_sorting = splitted_sorting
        self.method = params["method"]
        self.gt_sorting = params["method_kwargs"]["gt_sorting"]
        self.method_kwargs = params["method_kwargs"]
        self.result = {}

    def run(self, **job_kwargs):
        pass

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
        benchmark = MergingBenchmark(recording, gt_sorting, params)
        return benchmark