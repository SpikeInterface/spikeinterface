from __future__ import annotations

from spikeinterface.curation.auto_merge import auto_merge_units
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.widgets import (
    plot_unit_templates,
    plot_amplitudes,
    plot_crosscorrelograms,
)

import numpy as np
from .benchmark_base import Benchmark, BenchmarkStudy


class MergingBenchmark(Benchmark):

    def __init__(self, recording, splitted_sorting, params, gt_sorting, splitted_cells=None):
        self.recording = recording
        self.splitted_sorting = splitted_sorting
        self.gt_sorting = gt_sorting
        self.splitted_cells = splitted_cells
        self.method_kwargs = params["method_kwargs"]
        self.result = {}

    def run(self, **job_kwargs):
        sorting_analyzer = create_sorting_analyzer(
            self.splitted_sorting, self.recording, format="memory", sparse=True, **job_kwargs
        )
        # sorting_analyzer.compute(['random_spikes', 'templates'])
        # sorting_analyzer.compute('template_similarity', max_lag_ms=0.1, method="l2", **job_kwargs)
        merged_analyzer, self.result["merged_pairs"], self.result["merges"], self.result["outs"] = auto_merge_units(
            sorting_analyzer, extra_outputs=True, **self.method_kwargs, **job_kwargs
        )

        self.result["sorting"] = merged_analyzer.sorting

    def compute_result(self, **result_params):
        sorting = self.result["sorting"]
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=True)
        self.result["gt_comparison"] = comp

    _run_key_saved = [("sorting", "sorting"), ("merges", "pickle"), ("merged_pairs", "pickle"), ("outs", "pickle")]
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

    def get_count_units(self, case_keys=None, well_detected_score=None, redundant_score=None, overmerged_score=None):
        import pandas as pd

        if case_keys is None:
            case_keys = list(self.cases.keys())

        if isinstance(case_keys[0], str):
            index = pd.Index(case_keys, name=self.levels)
        else:
            index = pd.MultiIndex.from_tuples(case_keys, names=self.levels)

        columns = ["num_gt", "num_sorter", "num_well_detected"]
        comp = self.get_result(case_keys[0])["gt_comparison"]
        if comp.exhaustive_gt:
            columns.extend(["num_false_positive", "num_redundant", "num_overmerged", "num_bad"])
        count_units = pd.DataFrame(index=index, columns=columns, dtype=int)

        for key in case_keys:
            comp = self.get_result(key)["gt_comparison"]
            assert comp is not None, "You need to do study.run_comparisons() first"

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

    def plot_agreement_matrix(self, **kwargs):
        from .benchmark_plot_tools import plot_agreement_matrix

        return plot_agreement_matrix(self, **kwargs)

    def plot_unit_counts(self, case_keys=None, **kwargs):
        from .benchmark_plot_tools import plot_unit_counts

        return plot_unit_counts(self, case_keys, **kwargs)

    def get_splitted_pairs(self, case_key):
        return self.benchmarks[case_key].splitted_cells

    def get_splitted_pairs_index(self, case_key, pair):
        for count, i in enumerate(self.benchmarks[case_key].splitted_cells):
            if i == pair:
                return count

    def plot_splitted_amplitudes(self, case_key, pair_index=0, backend="ipywidgets"):
        analyzer = self.get_sorting_analyzer(case_key)
        if analyzer.get_extension("spike_amplitudes") is None:
            analyzer.compute(["spike_amplitudes"])
        plot_amplitudes(analyzer, unit_ids=self.get_splitted_pairs(case_key)[pair_index], backend=backend)

    def plot_splitted_correlograms(self, case_key, pair_index=0, backend="ipywidgets"):
        analyzer = self.get_sorting_analyzer(case_key)
        if analyzer.get_extension("correlograms") is None:
            analyzer.compute(["correlograms"])
        if analyzer.get_extension("template_similarity") is None:
            analyzer.compute(["template_similarity"])
        plot_crosscorrelograms(analyzer, unit_ids=self.get_splitted_pairs(case_key)[pair_index])

    def plot_splitted_templates(self, case_key, pair_index=0, backend="ipywidgets"):
        analyzer = self.get_sorting_analyzer(case_key)
        if analyzer.get_extension("spike_amplitudes") is None:
            analyzer.compute(["spike_amplitudes"])
        plot_unit_templates(analyzer, unit_ids=self.get_splitted_pairs(case_key)[pair_index], backend=backend)

    def plot_potential_merges(self, case_key, min_snr=None, backend="ipywidgets"):
        analyzer = self.get_sorting_analyzer(case_key)
        mylist = self.get_splitted_pairs(case_key)

        if analyzer.get_extension("spike_amplitudes") is None:
            analyzer.compute(["spike_amplitudes"])
        if analyzer.get_extension("correlograms") is None:
            analyzer.compute(["correlograms"])

        if min_snr is not None:
            select_from = analyzer.sorting.unit_ids
            if analyzer.get_extension("noise_levels") is None:
                analyzer.compute("noise_levels")
            if analyzer.get_extension("quality_metrics") is None:
                analyzer.compute("quality_metrics", metric_names=["snr"])

            snr = analyzer.get_extension("quality_metrics").get_data()["snr"].values
            select_from = select_from[snr > min_snr]
            mylist_selection = []
            for i in mylist:
                if (i[0] in select_from) or (i[1] in select_from):
                    mylist_selection += [i]
            mylist = mylist_selection

        from spikeinterface.widgets import plot_potential_merges

        plot_potential_merges(analyzer, mylist, backend=backend)

    def plot_performed_merges(self, case_key, backend="ipywidgets"):
        analyzer = self.get_sorting_analyzer(case_key)

        if analyzer.get_extension("spike_amplitudes") is None:
            analyzer.compute(["spike_amplitudes"])
        if analyzer.get_extension("correlograms") is None:
            analyzer.compute(["correlograms"])

        all_merges = list(self.benchmarks[case_key].result["merged_pairs"].values())

        from spikeinterface.widgets import plot_potential_merges

        plot_potential_merges(analyzer, all_merges, backend=backend)
