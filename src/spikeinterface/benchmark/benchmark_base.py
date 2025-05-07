from __future__ import annotations

from pathlib import Path
import shutil
import json
import numpy as np


import time


from spikeinterface.core import SortingAnalyzer
from spikeinterface.core.job_tools import fix_job_kwargs, split_job_kwargs
from spikeinterface import load, create_sorting_analyzer, load_sorting_analyzer
from spikeinterface.widgets import get_some_colors


import pickle

_key_separator = "_-°°-_"


class BenchmarkStudy:
    """
    Generic study for sorting components.
    This manage a list of Benchmark.
    This manage a dict of "cases" every case is one Benchmark.

    Benchmark is responsible for run() and compute_result()
    BenchmarkStudy is the main API for:
      * running (re-running) some cases
      * save (run + compute_result) in results dict
      * make some plots in inherited classes.


    """

    benchmark_class = None

    def __init__(self, study_folder):
        self.folder = Path(study_folder)
        self.datasets = {}
        self.analyzers = {}
        self.cases = {}
        self.benchmarks = {}
        self.levels = None
        self.colors_by_case = None
        self.colors_by_levels = {}
        self.scan_folder()

    @classmethod
    def create(cls, study_folder, datasets={}, cases={}, levels=None):
        """
        Create a BenchmarkStudy from a dict of datasets and cases.

        Parameters
        ----------
        study_folder : str | Path
            The folder where the study will be saved.
        datasets : dict
            A dict of datasets. The keys are the dataset names and the values are `SortingAnalyzer` objects.
            Values can also be tuples with (recording, gt_sorting), but this is deprecated.
        cases : dict
            A dict of cases. The keys are the cases (str, or tuples) and the values are dictionaries containing:

                * dataset
                * label
                * params
        levels : list | None
            If the keys of the cases are tuples, this is the list of levels names.

        Returns
        -------
        study : BenchmarkStudy
            The created study.
        """
        # check that cases keys are homogeneous
        key0 = list(cases.keys())[0]
        if isinstance(key0, str):
            assert all(isinstance(key, str) for key in cases.keys()), "Keys for cases are not homogeneous"
            if levels is None:
                levels = "level0"
            else:
                assert isinstance(levels, str)
        elif isinstance(key0, tuple):
            assert all(isinstance(key, tuple) for key in cases.keys()), "Keys for cases are not homogeneous"
            num_levels = len(key0)
            assert all(
                len(key) == num_levels for key in cases.keys()
            ), "Keys for cases are not homogeneous, tuple negth differ"
            if levels is None:
                levels = [f"level{i}" for i in range(num_levels)]
            else:
                levels = list(levels)
                assert len(levels) == num_levels
        else:
            raise ValueError("Keys for cases must str or tuple")

        study_folder = Path(study_folder)
        study_folder.mkdir(exist_ok=False, parents=True)

        # (study_folder / "datasets").mkdir()
        # (study_folder / "datasets" / "recordings").mkdir()
        # (study_folder / "datasets" / "gt_sortings").mkdir()
        (study_folder / "run_logs").mkdir()
        # (study_folder / "metrics").mkdir()
        (study_folder / "results").mkdir()
        (study_folder / "sorting_analyzer").mkdir()

        analyzers_path = {}
        # for key, (rec, gt_sorting) in datasets.items():
        for key, data in datasets.items():
            assert "/" not in key, "'/' cannot be in the key name!"
            assert "\\" not in key, "'\\' cannot be in the key name!"

            local_analyzer_folder = study_folder / "sorting_analyzer" / key

            if isinstance(data, tuple):
                # old case : rec + sorting
                rec, gt_sorting = data
                analyzer = create_sorting_analyzer(
                    gt_sorting, rec, sparse=True, format="binary_folder", folder=local_analyzer_folder
                )
                analyzer.compute("random_spikes")
                analyzer.compute("templates")
                analyzer.compute("noise_levels")
            else:
                # new case : analzyer
                assert isinstance(data, SortingAnalyzer)
                analyzer = data
                if data.format == "memory":
                    # then copy a local copy in the folder
                    analyzer = data.save_as(format="binary_folder", folder=local_analyzer_folder)
                else:
                    analyzer = data

                rec, gt_sorting = analyzer.recording, analyzer.sorting

            analyzers_path[key] = str(analyzer.folder.resolve())

            # recordings are pickled
            # rec.dump_to_pickle(study_folder / f"datasets/recordings/{key}.pickle")

            # sortings are pickled + saved as NumpyFolderSorting
            # gt_sorting.dump_to_pickle(study_folder / f"datasets/gt_sortings/{key}.pickle")
            # gt_sorting.save(format="numpy_folder", folder=study_folder / f"datasets/gt_sortings/{key}")

        # analyzer path (local or external)
        (study_folder / "analyzers_path.json").write_text(json.dumps(analyzers_path, indent=4), encoding="utf8")

        info = {}
        info["levels"] = levels
        (study_folder / "info.json").write_text(json.dumps(info, indent=4), encoding="utf8")

        # cases is dumped to a pickle file, json is not possible because of the tuple key
        (study_folder / "cases.pickle").write_bytes(pickle.dumps(cases))

        return cls(study_folder)

    def create_benchmark(self, key):
        """
        Create a benchmark for a given key.
        """
        raise NotImplementedError

    def scan_folder(self):
        if not (self.folder / "sorting_analyzer").exists():
            raise ValueError(f"This is folder is not a BenchmarkStudy : {self.folder.absolute()}")

        with open(self.folder / "info.json", "r") as f:
            self.info = json.load(f)

        with open(self.folder / "analyzers_path.json", "r") as f:
            self.analyzers_path = json.load(f)

        self.levels = self.info["levels"]

        for key, folder in self.analyzers_path.items():
            analyzer = load_sorting_analyzer(folder, load_extensions=False)
            self.analyzers[key] = analyzer
            # the sorting is in memory here we take the saved one because comparisons need to pickle it later
            sorting = load(analyzer.folder / "sorting")
            self.datasets[key] = analyzer.recording, sorting

        with open(self.folder / "cases.pickle", "rb") as f:
            self.cases = pickle.load(f)

        self.benchmarks = {}
        for key in self.cases:
            result_folder = self.folder / "results" / self.key_to_str(key)
            if result_folder.exists():
                result = self.benchmark_class.load_folder(result_folder)
                benchmark = self.create_benchmark(key=key)
                benchmark.result.update(result)
                self.benchmarks[key] = benchmark
            else:
                self.benchmarks[key] = None

    def __repr__(self):
        t = f"{self.__class__.__name__} {self.folder.stem} \n"
        t += f"  datasets: {len(self.datasets)} {list(self.datasets.keys())}\n"
        t += f"  cases: {len(self.cases)} {list(self.cases.keys())}\n"
        num_computed = sum([1 for benchmark in self.benchmarks.values() if benchmark is not None])
        t += f"  computed: {num_computed}\n"
        return t

    def key_to_str(self, key):
        if isinstance(key, str):
            return key
        elif isinstance(key, tuple):
            return _key_separator.join([str(k) for k in key])
        else:
            raise ValueError("Keys for cases must str or tuple")

    def remove_benchmark(self, key):
        result_folder = self.folder / "results" / self.key_to_str(key)
        log_file = self.folder / "run_logs" / f"{self.key_to_str(key)}.json"

        if result_folder.exists():
            shutil.rmtree(result_folder)
        for f in (log_file,):
            if f.exists():
                f.unlink()
        self.benchmarks[key] = None

    def run(self, case_keys=None, keep=True, verbose=False, **job_kwargs):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        job_keys = []
        for key in case_keys:

            result_folder = self.folder / "results" / self.key_to_str(key)
            sorter_folder = self.folder / "sorters" / self.key_to_str(key)

            if keep and result_folder.exists():
                continue
            elif not keep and (result_folder.exists() or sorter_folder.exists()):
                self.remove_benchmark(key)
            job_keys.append(key)

        for key in job_keys:
            benchmark = self.create_benchmark(key)
            t0 = time.perf_counter()
            benchmark.run(**job_kwargs)
            t1 = time.perf_counter()
            self.benchmarks[key] = benchmark
            bench_folder = self.folder / "results" / self.key_to_str(key)
            bench_folder.mkdir(exist_ok=True)
            benchmark.save_run(bench_folder)
            benchmark.result["run_time"] = float(t1 - t0)
            benchmark.save_main(bench_folder)

    def set_colors(self, colors=None, map_name="tab10", levels_to_group_by=None):
        """
        Set colors for the study cases or for a given levels_to_group_by.

        Parmeters
        ---------
        colors : dict | None, default: None
            A user-defined dictionary with the case keys as keys and the colors as values.
            Note that the case keys depend on the levels_to_group_by.
        map_name : str, default: 'tab10'
            The name of the colormap to use.
        levels_to_group_by : list | None, default: None
            The levels to group by. If None, the colors are set for the cases.
        """
        case_keys, _ = self.get_grouped_keys_mapping(levels_to_group_by)

        if colors is None:
            colors = get_some_colors(
                case_keys, map_name=map_name, color_engine="matplotlib", shuffle=False, margin=0, resample=False
            )
            if levels_to_group_by is None:
                self.colors_by_case = colors
            else:
                level_key = tuple(levels_to_group_by) if len(levels_to_group_by) > 1 else levels_to_group_by[0]
                self.colors_by_levels[level_key] = colors
        else:
            assert all([key in colors for key in case_keys]), f"You must provide colors for all cases keys: {case_keys}"
            if levels_to_group_by is None:
                self.colors_by_case = colors
            else:
                level_key = tuple(levels_to_group_by) if len(levels_to_group_by) > 1 else levels_to_group_by[0]
                self.colors_by_levels[level_key] = colors

    def get_colors(self, levels_to_group_by=None):
        if levels_to_group_by is None:
            if self.colors_by_case is None:
                self.set_colors()
            return self.colors_by_case
        else:
            level_key = tuple(levels_to_group_by) if len(levels_to_group_by) > 1 else levels_to_group_by[0]
            if level_key not in self.colors_by_levels:
                self.set_colors(levels_to_group_by=levels_to_group_by)
            return self.colors_by_levels[level_key]

    def get_run_times(self, case_keys=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        run_times = {}
        for key in case_keys:
            benchmark = self.benchmarks[key]
            assert benchmark is not None
            run_times[key] = benchmark.result["run_time"]
        import pandas as pd

        df = pd.DataFrame(dict(run_times=run_times))
        if not isinstance(self.levels, str):
            df.index.names = self.levels
        return df

    def get_grouped_keys_mapping(self, levels_to_group_by=None):
        """
        Return a dictionary of grouped keys.

        Parameters
        ----------
        levels_to_group_by : list
            A list of levels to group by.

        Returns
        -------
        grouped_keys : dict
            A dictionary of grouped keys, with the new keys as keys and the list of cases
            associated to new keys as values.
        labels : dict
            A dictionary of labels, with the new keys as keys and the labels as values.
        """
        cases = list(self.cases.keys())
        if levels_to_group_by is None or self.levels is None:
            keys_mapping = {key: [key] for key in cases}
        elif len(self.levels) == 1:
            keys_mapping = {key: [key] for key in cases}
        else:
            study_levels = self.levels
            assert np.all(
                [l in study_levels for l in levels_to_group_by]
            ), f"levels_to_group_by must be in {study_levels}, got {levels_to_group_by}"
            keys_mapping = {}
            for key in cases:
                new_key = tuple(key[list(study_levels).index(level)] for level in levels_to_group_by)
                if len(new_key) == 1:
                    new_key = new_key[0]
                if new_key not in keys_mapping:
                    keys_mapping[new_key] = []
                keys_mapping[new_key].append(key)

        if levels_to_group_by is None:
            labels = {key: self.cases[key]["label"] for key in cases}
        else:
            key0 = list(keys_mapping.keys())[0]
            if isinstance(key0, tuple):
                labels = {key: "-".join(key) for key in keys_mapping}
            else:
                labels = {key: key for key in keys_mapping}

        return keys_mapping, labels

    def plot_run_times(self, case_keys=None, **kwargs):
        from .benchmark_plot_tools import plot_run_times

        return plot_run_times(self, case_keys=case_keys, **kwargs)

    def compute_results(self, case_keys=None, verbose=False, **result_params):

        if case_keys is None:
            case_keys = list(self.cases.keys())

        job_keys = []
        for key in case_keys:
            benchmark = self.benchmarks[key]
            assert benchmark is not None
            benchmark.compute_result(**result_params)
            benchmark.save_result(self.folder / "results" / self.key_to_str(key))

    def create_sorting_analyzer_gt(self, case_keys=None, return_scaled=True, random_params={}, **job_kwargs):
        print("###### Study.create_sorting_analyzer_gt() is not used anymore!!!!!!")
        # if case_keys is None:
        #     case_keys = self.cases.keys()

        # base_folder = self.folder / "sorting_analyzer"
        # base_folder.mkdir(exist_ok=True)

        # dataset_keys = [self.cases[key]["dataset"] for key in case_keys]
        # dataset_keys = set(dataset_keys)
        # for dataset_key in dataset_keys:
        #     # the waveforms depend on the dataset key
        #     folder = base_folder / self.key_to_str(dataset_key)
        #     recording, gt_sorting = self.datasets[dataset_key]
        #     sorting_analyzer = create_sorting_analyzer(
        #         gt_sorting, recording, format="binary_folder", folder=folder, return_scaled=return_scaled
        #     )
        #     sorting_analyzer.compute("random_spikes", **random_params)
        #     sorting_analyzer.compute("templates", **job_kwargs)
        #     sorting_analyzer.compute("noise_levels")

    def get_sorting_analyzer(self, case_key=None, dataset_key=None):
        if case_key is not None:
            dataset_key = self.cases[case_key]["dataset"]
        return self.analyzers[dataset_key]

        # folder = self.folder / "sorting_analyzer" / self.key_to_str(dataset_key)
        # sorting_analyzer = load_sorting_analyzer(folder)
        # return sorting_analyzer

    def compute_analyzer_extension(self, extensions, dataset_keys=None, **extension_kwargs):
        if dataset_keys is None:
            dataset_keys = list(self.datasets.keys())
        if not isinstance(dataset_keys, list):
            dataset_keys = [dataset_keys]
        for dataset_key in dataset_keys:
            sorting_analyzer = self.get_sorting_analyzer(dataset_key=dataset_key)
            sorting_analyzer.compute(extensions, **extension_kwargs)

    def get_gt_unit_locations(self, case_key):
        dataset_key = self.cases[case_key]["dataset"]
        sorting_analyzer = self.get_sorting_analyzer(dataset_key=dataset_key)
        if "gt_unit_locations" in sorting_analyzer.sorting.get_property_keys():
            return sorting_analyzer.get_sorting_property("gt_unit_locations")
        else:
            if not sorting_analyzer.has_extension("unit_locations"):
                self.compute_analyzer_extension(["unit_locations"], dataset_keys=dataset_key)
            unit_locations_ext = sorting_analyzer.get_extension("unit_locations")
            return unit_locations_ext.get_data()

    def get_templates(self, key, operator="average"):
        sorting_analyzer = self.get_sorting_analyzer(case_key=key)
        templates = sorting_analyzer.get_extenson("templates").get_data(operator=operator)
        return templates

    def compute_metrics(self, case_keys=None, metric_names=["snr", "firing_rate"], force=False, **job_kwargs):
        if case_keys is None:
            case_keys = self.cases.keys()

        done = []
        for key in case_keys:
            dataset_key = self.cases[key]["dataset"]
            if dataset_key in done:
                # some case can share the same analyzer
                continue
            done.append(dataset_key)
            # filename = self.folder / "metrics" / f"{self.key_to_str(dataset_key)}.csv"
            # if filename.exists():
            #     if force:
            #         os.remove(filename)
            #     else:
            #         continue
            sorting_analyzer = self.get_sorting_analyzer(key)
            qm_ext = sorting_analyzer.get_extension("quality_metrics")
            if qm_ext is None or force:
                qm_ext = sorting_analyzer.compute("quality_metrics", metric_names=metric_names, **job_kwargs)

            # TODO remove this metics CSV file!!!!
            metrics = qm_ext.get_data()
            # metrics.to_csv(filename, sep="\t", index=True)

    def get_metrics(self, key):
        analyzer = self.get_sorting_analyzer(key)
        ext = analyzer.get_extension("quality_metrics")
        if ext is None:
            # TODO au to compute ????
            return None

        metrics = ext.get_data()
        # add GT unit id column
        gt_unit_ids = analyzer.sorting.unit_ids
        metrics.loc[:, "gt_unit_id"] = gt_unit_ids
        return metrics

    def get_all_metrics(self, case_keys=None):
        """
        Return a DataFrame with concatented metrics for multiple cases.
        """
        import pandas as pd

        if case_keys is None:
            case_keys = list(self.cases.keys())
        assert all(key in self.cases for key in case_keys), "Some case keys are not in cases"
        metrics = []
        indices = []
        for key in case_keys:
            metrics.append(self.get_metrics(key))
            indices.extend([key] * len(metrics[-1]))
        if isinstance(case_keys[0], str):
            index = pd.Index(indices, name=self.levels)
        else:
            index = pd.MultiIndex.from_tuples(indices, names=self.levels)
        metrics = pd.concat(metrics)
        metrics.index = index
        return metrics

    def get_units_snr(self, key):
        """ """
        return self.get_metrics(key)["snr"]

    def get_result(self, key):
        return self.benchmarks[key].result

    def get_pairs_by_level(self, level):
        """
        usefull for function like plot_performance_losses() where you need to plot one pair of results
        This generate list of pairs for a given level.
        """

        level_index = self.levels.index(level)

        possible_values = []
        for key in self.cases.keys():
            assert isinstance(key, tuple), "get_pairs_by_level need tuple keys"
            level_value = key[level_index]
            if level_value not in possible_values:
                possible_values.append(level_value)
        assert len(possible_values) == 2, "get_pairs_by_level() : you need exactly 2 value for this levels"

        pairs = []
        for key in self.cases.keys():

            case0 = list(key)
            case1 = list(key)
            case0[level_index] = possible_values[0]
            case1[level_index] = possible_values[1]
            case0 = tuple(case0)
            case1 = tuple(case1)

            pair = (case0, case1)

            if pair not in pairs:
                pairs.append(pair)

        return pairs


class Benchmark:
    """
    Responsible to make a unique run() and compute_result() for one case.
    """

    def __init__(self):
        self.result = {}

    # this must not be changed in inherited
    _main_key_saved = [
        ("run_time", "pickle"),
    ]
    # this must be updated in hirerited
    _run_key_saved = []
    _result_key_saved = []

    def _save_keys(self, saved_keys, folder):
        for k, format in saved_keys:
            if k not in self.result or self.result[k] is None:
                continue
            if format == "npy":
                np.save(folder / f"{k}.npy", self.result[k])
            elif format == "pickle":
                with open(folder / f"{k}.pickle", mode="wb") as f:
                    pickle.dump(self.result[k], f)
            elif format == "sorting":
                self.result[k].save(folder=folder / k, format="numpy_folder", overwrite=True)
            elif format == "Motion":
                self.result[k].save(folder=folder / k)
            elif format == "zarr_templates":
                self.result[k].to_zarr(folder / k)
            elif format == "sorting_analyzer":
                pass
            else:
                raise ValueError(f"Save error {k} {format}")

    def save_main(self, folder):
        # used for run time
        self._save_keys(self._main_key_saved, folder)

    def save_run(self, folder):
        self._save_keys(self._run_key_saved, folder)

    def save_result(self, folder):
        self._save_keys(self._result_key_saved, folder)

    @classmethod
    def load_folder(cls, folder):
        result = {}
        for k, format in cls._run_key_saved + cls._result_key_saved + cls._main_key_saved:
            if format == "npy":
                file = folder / f"{k}.npy"
                if file.exists():
                    result[k] = np.load(file)
            elif format == "pickle":
                file = folder / f"{k}.pickle"
                if file.exists():
                    with open(file, mode="rb") as f:
                        result[k] = pickle.load(f)
            elif format == "sorting":
                from spikeinterface.core import load_extractor

                result[k] = load(folder / k)
            elif format == "Motion":
                from spikeinterface.core.motion import Motion

                result[k] = Motion.load(folder / k)
            elif format == "zarr_templates":
                from spikeinterface.core.template import Templates

                result[k] = Templates.from_zarr(folder / k)

        return result

    def run(self):
        # run method
        raise NotImplementedError

    def compute_result(self):
        # run becnhmark result
        raise NotImplementedError


# Common feature accross some benchmark : sorter + matching
class MixinStudyUnitCount:
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
