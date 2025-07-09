"""
This replace the previous `GroundTruthStudy`
"""

import numpy as np
from spikeinterface.core import NumpySorting, create_sorting_analyzer
from .benchmark_base import Benchmark, BenchmarkStudy, MixinStudyUnitCount
from spikeinterface.sorters import run_sorter
from spikeinterface.comparison import compare_multiple_sorters

from spikeinterface.benchmark import analyse_residual


# TODO later integrate CollisionGTComparison optionally in this class.


class SorterBenchmarkWithoutGroundTruth(Benchmark):
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

    def compute_result(self, residulal_peak_threshold=6, **job_kwargs):

        sorting = self.result["sorting"]
        analyzer = create_sorting_analyzer(sorting, self.recording, sparse=True, format="memory", **job_kwargs)
        analyzer.compute("random_spikes")
        analyzer.compute("templates")
        analyzer.compute("noise_levels")
        analyzer.compute({"spike_amplitudes": {}, "amplitude_scalings": {"handle_collisions": False}}, **job_kwargs)

        analyzer.compute("quality_metrics", **job_kwargs)

        residual, peaks = analyse_residual(
            analyzer,
            detect_peaks_kwargs=dict(
                method="locally_exclusive",
                peak_sign="neg",
                detect_threshold=residulal_peak_threshold,
            ),
            **job_kwargs,
        )

        self.result["sorter_analyzer"] = analyzer
        self.result["peaks_from_residual"] = peaks

    _run_key_saved = [
        ("sorting", "sorting"),
    ]
    _result_key_saved = [
        # note that this multi_comp is the same accros benchmark (cases)
        ("multi_comp", "pickle"),
        ("sorter_analyzer", "sorting_analyzer"),
        ("peaks_from_residual", "npy"),
    ]


class SorterStudyWithoutGroundTruth(BenchmarkStudy):
    """
    This class is an alternative to SorterStudy when the dataset do not have groundtruth.

    This is mainly base on the residual analysis.
    """

    benchmark_class = SorterBenchmarkWithoutGroundTruth

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        sorter_folder = self.folder / "sorters" / self.key_to_str(key)
        benchmark = SorterBenchmarkWithoutGroundTruth(recording, gt_sorting, params, sorter_folder)
        return benchmark

    def _get_comparison_groups(self):
        # multicomparison are done on all cases sharing the same dataset key.
        case_keys = list(self.cases.keys())
        groups = {}
        for case_key in case_keys:
            data_key = self.cases[case_key]["dataset"]
            if data_key not in groups:
                groups[data_key] = []
            groups[data_key].append(case_key)
        return groups

    def compute_results(
        self, case_keys=None, verbose=False, delta_time=0.4, match_score=0.5, chance_score=0.1, **result_params
    ):
        # Here we need a hack because the results is not computed case by case but all at once

        assert case_keys is None, "SorterStudyWithoutGroundTruth do not permit compute_results for sub cases"

        # allways the full list
        case_keys = list(self.cases.keys())

        # First : this do the case by case internally SorterBenchmarkWithoutGroundTruth.compute_result()
        BenchmarkStudy.compute_results(self, case_keys=case_keys, verbose=verbose, **result_params)

        # Then we need to compute the multicomparison for case that have the same dataset key.
        groups = self._get_comparison_groups()

        for data_key, group in groups.items():

            sorting_list = [self.get_result(key)["sorting"] for key in group]
            name_list = [key for key in group]
            multi_comp = compare_multiple_sorters(
                sorting_list,
                name_list=name_list,
                delta_time=delta_time,
                match_score=0.5,
                chance_score=0.1,
                agreement_method="count",
                n_jobs=-1,
                spiketrain_mode="union",
                verbose=verbose,
                do_matching=True,
            )
            # and then the same multi comp is stored for each case_key
            for key in case_keys:
                benchmark = self.benchmarks[key]
                benchmark.result["multi_comp"] = multi_comp
                benchmark.save_result(self.folder / "results" / self.key_to_str(key))

    def plot_residual_peak_amplitudes(self, figsize=None):
        import matplotlib.pyplot as plt

        groups = self._get_comparison_groups()
        colors = self.get_colors()

        for data_key, group in groups.items():
            fig, ax = plt.subplots(figsize=figsize)

            lim0, lim1 = np.inf, -np.inf

            for key in group:
                peaks = self.get_result(key)["peaks_from_residual"]

                lim0 = min(lim0, np.min(peaks["amplitude"]))
                lim1 = max(lim1, np.max(peaks["amplitude"]))

            bins = np.linspace(lim0, lim1, 200)
            if lim1 < 0:
                lim1 = 0
            if lim0 > 0:
                lim0 = 0

            for key in group:
                peaks = self.get_result(key)["peaks_from_residual"]
                print(peaks.size)
                print()
                count, bins = np.histogram(peaks["amplitude"], bins=bins)
                ax.plot(bins[:-1], count, color=colors[key], label=self.cases[key]["label"])

            ax.legend()

    # def plot_quality_metrics_comparison_on_agreement(self, qm_name='rp_contamination', figsize=None):
    #     import matplotlib.pyplot as plt

    #     groups = self._get_comparison_groups()

    #     for data_key, group in groups.items():
    #         n = len(group)
    #         fig, axs = plt.subplots(ncols=n - 1, nrows=n - 1, figsize=figsize, squeeze=False)
    #         for i, key1 in enumerate(group):
    #             for j, key2 in enumerate(group):
    #                 if i < j:
    #                     ax = axs[i, j - 1]
    #                     label1 = self.cases[key1]['label']
    #                     label2 = self.cases[key2]['label']

    #                     if i == j - 1:
    #                         ax.set_xlabel(label2)
    #                         ax.set_ylabel(label1)

    #                     multi_comp = self.get_result(key1)['multi_comp']
    #                     comp = multi_comp.comparisons[key1, key2]

    #                     match_12 = comp.hungarian_match_12
    #                     if match_12.dtype.kind =='i':
    #                         mask = match_12.values != -1
    #                     if match_12.dtype.kind =='U':
    #                         mask = match_12.values != ''

    #                     common_unit1_ids = match_12[mask].index
    #                     common_unit2_ids = match_12[mask].values
    #                     metrics1 = self.get_result(key1)["sorter_analyzer"].get_extension("quality_metrics").get_data()
    #                     metrics2 = self.get_result(key2)["sorter_analyzer"].get_extension("quality_metrics").get_data()

    #                     values1 = metrics1.loc[common_unit1_ids, qm_name].values
    #                     values2 = metrics2.loc[common_unit2_ids, qm_name].values

    #                     print(common_unit1_ids, metrics1.columns, values1)
    #                     print(common_unit2_ids, metrics2.columns, values2)

    #                     ax.scatter(values1, values2)
    #                     if i != j - 1:
    #                         ax.set_xlabel("")
    #                         ax.set_ylabel("")
    #                         ax.set_xticks([])
    #                         ax.set_yticks([])
    #                         ax.set_xticklabels([])
    #                         ax.set_yticklabels([])
