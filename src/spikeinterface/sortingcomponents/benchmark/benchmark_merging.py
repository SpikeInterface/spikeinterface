from __future__ import annotations

from spikeinterface.sortingcomponents.merging import merge_spikes
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spikeinterface.widgets import (
    plot_agreement_matrix,
    plot_unit_templates,
    plot_amplitudes,
    plot_crosscorrelograms,
)

import numpy as np
from spikeinterface.sortingcomponents.benchmark.benchmark_tools import Benchmark, BenchmarkStudy


class MergingBenchmark(Benchmark):

    def __init__(self, recording, splitted_sorting, params, gt_sorting, splitted_cells=None):
        self.recording = recording
        self.splitted_sorting = splitted_sorting
        self.method = params["method"]
        self.gt_sorting = gt_sorting
        self.splitted_cells = splitted_cells
        self.method_kwargs = params["method_kwargs"]
        self.result = {}

    def run(self, **job_kwargs):
        self.result["sorting"], self.result["merges"] = merge_spikes(
            self.recording,
            self.splitted_sorting,
            method=self.method,
            method_kwargs=self.method_kwargs,
            extra_outputs=True,
        )

    def compute_result(self, **result_params):
        sorting = self.result["sorting"]
        comp = compare_sorter_to_ground_truth(self.gt_sorting, sorting, exhaustive_gt=True)
        self.result["gt_comparison"] = comp

    _run_key_saved = [("sorting", "sorting"), ("merges", "pickle")]
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

    def plot_agreements(self, case_keys=None, figsize=(15, 15)):
        import matplotlib.pyplot as plt

        if case_keys is None:
            case_keys = list(self.cases.keys())

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)
        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            ax.set_title(self.cases[key]["label"])
            plot_agreement_matrix(self.get_result(key)["gt_comparison"], ax=ax)

        return fig

    def plot_unit_counts(self, case_keys=None, figsize=None, **extra_kwargs):
        from spikeinterface.widgets.widget_list import plot_study_unit_counts

        plot_study_unit_counts(self, case_keys, figsize=figsize, **extra_kwargs)

    def get_splitted_pairs(self, case_key):
        return self.benchmarks[case_key].splitted_cells

    def plot_splitted_amplitudes(self, case_key, pair_index=0):
        analyzer = self.get_sorting_analyzer(case_key)
        if analyzer.get_extension("spike_amplitudes") is None:
            analyzer.compute(["spike_amplitudes"])
        plot_amplitudes(analyzer, unit_ids=self.get_splitted_pairs(case_key)[pair_index])

    def plot_splitted_correlograms(self, case_key, pair_index=0):
        analyzer = self.get_sorting_analyzer(case_key)
        if analyzer.get_extension("correlograms") is None:
            analyzer.compute(["correlograms"])
        if analyzer.get_extension("template_similarity") is None:
            analyzer.compute(["template_similarity"])
        plot_crosscorrelograms(analyzer, unit_ids=self.get_splitted_pairs(case_key)[pair_index])

    def plot_splitted_templates(self, case_key, pair_index=0):
        analyzer = self.get_sorting_analyzer(case_key)
        if analyzer.get_extension("spike_amplitudes") is None:
            analyzer.compute(["spike_amplitudes"])
        plot_unit_templates(analyzer, unit_ids=self.get_splitted_pairs(case_key)[pair_index])

    # def visualize_splits(self, case_key, figsize=(15, 5)):
    #     cc_similarities = []
    #     from spikeinterface.curation. import compute_presence_distance

    #     analyzer = self.get_sorting_analyzer(case_key)
    #     if analyzer.get_extension("template_similarity") is None:
    #         analyzer.compute(["template_similarity"])

    #     distances = {}
    #     distances["similarity"] = analyzer.get_extension("template_similarity").get_data()
    #     sorting = analyzer.sorting

    #     distances["time_distance"] = np.ones((analyzer.get_num_units(), analyzer.get_num_units()))
    #     for i, unit1 in enumerate(analyzer.unit_ids):
    #         for j, unit2 in enumerate(analyzer.unit_ids):
    #             if unit2 <= unit1:
    #                 continue
    #             d = compute_presence_distance(analyzer, unit1, unit2)
    #             distances["time_distance"][i, j] = d

    #     import lussac.utils as utils

    #     distances["cross_cont"] = np.ones((analyzer.get_num_units(), analyzer.get_num_units()))
    #     for i, unit1 in enumerate(analyzer.unit_ids):
    #         for j, unit2 in enumerate(analyzer.unit_ids):
    #             if unit2 <= unit1:
    #                 continue
    #             spike_train1 = np.array(sorting.get_unit_spike_train(unit1))
    #             spike_train2 = np.array(sorting.get_unit_spike_train(unit2))
    #             distances["cross_cont"][i, j], _ = utils.estimate_cross_contamination(
    #                 spike_train1, spike_train2, (1, 4), limit=0.1
    #             )

    #     splits = np.array(self.benchmarks[case_key].splitted_cells)
    #     src, tgt = splits[:, 0], splits[:, 1]
    #     src = analyzer.sorting.ids_to_indices(src)
    #     tgt = analyzer.sorting.ids_to_indices(tgt)
    #     import matplotlib.pyplot as plt

    #     fig, axs = plt.subplots(ncols=2, nrows=2, figsize=figsize, squeeze=True)
    #     axs[0, 0].scatter(distances["similarity"].flatten(), distances["time_distance"].flatten(), c="k", alpha=0.25)
    #     axs[0, 0].scatter(distances["similarity"][src, tgt], distances["time_distance"][src, tgt], c="r")
    #     axs[0, 0].set_xlabel("cc similarity")
    #     axs[0, 0].set_ylabel("presence ratio")

    #     axs[1, 0].scatter(distances["similarity"].flatten(), distances["cross_cont"].flatten(), c="k", alpha=0.25)
    #     axs[1, 0].scatter(distances["similarity"][src, tgt], distances["cross_cont"][src, tgt], c="r")
    #     axs[1, 0].set_xlabel("cc similarity")
    #     axs[1, 0].set_ylabel("cross cont")

    #     axs[0, 1].scatter(distances["cross_cont"].flatten(), distances["time_distance"].flatten(), c="k", alpha=0.25)
    #     axs[0, 1].scatter(distances["cross_cont"][src, tgt], distances["time_distance"][src, tgt], c="r")
    #     axs[0, 1].set_xlabel("cross_cont")
    #     axs[0, 1].set_ylabel("presence ratio")

    #     plt.show()
