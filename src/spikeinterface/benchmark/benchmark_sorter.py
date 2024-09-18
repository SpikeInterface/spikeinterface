"""
This replace the previous `GroundTruthStudy`
"""


import numpy as np
from ..core import NumpySorting
from .benchmark_base import Benchmark, BenchmarkStudy
from ..sorters import run_sorter
from spikeinterface.comparison import compare_sorter_to_ground_truth

# from spikeinterface.widgets import (
#     plot_agreement_matrix,
#     plot_comparison_collision_by_similarity,
# )





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

    def compute_result(self):
        # run becnhmark result
        sorting = self.result["sorting"]
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=True)
        self.result["gt_comparison"] = comp

    _run_key_saved = [
        ("sorting", "sorting"),
    ]
    _result_key_saved = [
        ("gt_comparison", "pickle"),
    ]

class SorterStudy(BenchmarkStudy):
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

    def get_performance_by_unit(self, case_keys=None):
        import pandas as pd

        if case_keys is None:
            case_keys = self.cases.keys()

        perf_by_unit = []
        for key in case_keys:
            comp = self.get_result(key)["gt_comparison"]

            perf = comp.get_performance(method="by_unit", output="pandas")

            if isinstance(key, str):
                perf[self.levels] = key
            elif isinstance(key, tuple):
                for col, k in zip(self.levels, key):
                    perf[col] = k

            perf = perf.reset_index()
            perf_by_unit.append(perf)

        perf_by_unit = pd.concat(perf_by_unit)
        perf_by_unit = perf_by_unit.set_index(self.levels)
        perf_by_unit = perf_by_unit.sort_index()
        return perf_by_unit

    def get_count_units(self, case_keys=None, well_detected_score=None, redundant_score=None, overmerged_score=None):
        import pandas as pd

        if case_keys is None:
            case_keys = list(self.cases.keys())

        if isinstance(case_keys[0], str):
            index = pd.Index(case_keys, name=self.levels)
        else:
            index = pd.MultiIndex.from_tuples(case_keys, names=self.levels)

        columns = ["num_gt", "num_sorter", "num_well_detected"]
        key0 = case_keys[0]
        comp = self.get_result(key0)["gt_comparison"]
        if comp.exhaustive_gt:
            columns.extend(["num_false_positive", "num_redundant", "num_overmerged", "num_bad"])
        count_units = pd.DataFrame(index=index, columns=columns, dtype=int)

        for key in case_keys:
            comp = self.get_result(key)["gt_comparison"]

            gt_sorting = comp.sorting1
            sorting = comp.sorting2

            count_units.loc[key, "num_gt"] = len(gt_sorting.get_unit_ids())
            count_units.loc[key, "num_sorter"] = len(sorting.get_unit_ids())
            count_units.loc[key, "num_well_detected"] = comp.count_well_detected_units(well_detected_score)

            if comp.exhaustive_gt:
                count_units.loc[key, "num_redundant"] = comp.count_redundant_units(redundant_score)
                count_units.loc[key, "num_overmerged"] = comp.count_overmerged_units(overmerged_score)
                count_units.loc[key, "num_false_positive"] = comp.count_false_positive_units(redundant_score)
                count_units.loc[key, "num_bad"] = comp.count_bad_units()

        return count_units

    # plotting as methods
    def plot_unit_counts(self, **kwargs):
        from .benchmark_plot_tools import plot_unit_counts
        return plot_unit_counts(self, **kwargs)

    def plot_performances(self, **kwargs):
        from .benchmark_plot_tools import plot_performances
        return plot_performances(self, **kwargs)

    def plot_agreement_matrix(self, **kwargs):
        from .benchmark_plot_tools import plot_agreement_matrix
        return plot_agreement_matrix(self, **kwargs)



