"""
This replace the previous `GroundTruthStudy`
"""

import numpy as np
from spikeinterface.core import NumpySorting
from .benchmark_base import Benchmark, BenchmarkStudy, MixinStudyUnitCount
from spikeinterface.sorters import run_sorter
from spikeinterface.comparison import compare_multiple_sorters




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

    def compute_result(self, exhaustive_gt=True):
        # TODO
        pass

    _run_key_saved = [
        ("sorting", "sorting"),
    ]
    _result_key_saved = [
        ("multi_comp", "pickle"),
    ]



class SorterStudyWithoutGroundTruth(BenchmarkStudy):
    """
    This class is an alternative to SorterStudy when the dataset do not have groundtruth
    """

    benchmark_class = SorterBenchmarkWithoutGroundTruth

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        sorter_folder = self.folder / "sorters" / self.key_to_str(key)
        benchmark = SorterBenchmarkWithoutGroundTruth(recording, gt_sorting, params, sorter_folder)
        return benchmark
    
    def compute_results(self, case_keys=None, verbose=False, delta_time=0.4, match_score=0.5, chance_score=0.1, **result_params):
        # Here we need a hack because the results is not computed case by case but all at once

        assert case_keys is None, "SorterStudyWithoutGroundTruth do not permit compute_results for sub cases"

        # allways the full list
        case_keys = list(self.cases.keys())

        # First : this do the case by case internally SorterBenchmarkWithoutGroundTruth.compute_result()
        BenchmarkStudy.compute_results(self, case_keys=case_keys, verbose=verbose, **result_params)

        # Then we need to compute the multicomparison for case that have the same dataset key.
        groups = {}
        for case_key in case_keys:
            data_key = self.cases[case_key]['dataset']
            if data_key not in groups:
                groups[data_key] = []
            groups[data_key].append(case_key)
        
        for data_key, group in groups.items():

            sorting_list = [self.get_result(key)['sorting'] for key in group]
            multi_comp = compare_multiple_sorters(
                sorting_list,
                name_list=None,
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
                benchmark.result['multi_comp'] = multi_comp
                benchmark.save_result(self.folder / "results" / self.key_to_str(key))

