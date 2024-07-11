from __future__ import annotations

from pathlib import Path
import shutil
import json
import numpy as np


import time


from spikeinterface.core import SortingAnalyzer

from spikeinterface import load_extractor, create_sorting_analyzer, load_sorting_analyzer
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
        self.scan_folder()
        self.colors = None

    @classmethod
    def create(cls, study_folder, datasets={}, cases={}, levels=None):
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

    def create_benchmark(self):
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
            analyzer = load_sorting_analyzer(folder)
            self.analyzers[key] = analyzer
            # the sorting is in memory here we take the saved one because comparisons need to pickle it later
            sorting = load_extractor(analyzer.folder / "sorting")
            self.datasets[key] = analyzer.recording, sorting

        # for rec_file in (self.folder / "datasets" / "recordings").glob("*.pickle"):
        #     key = rec_file.stem
        #     rec = load_extractor(rec_file)
        #     gt_sorting = load_extractor(self.folder / f"datasets" / "gt_sortings" / key)
        #     self.datasets[key] = (rec, gt_sorting)

        with open(self.folder / "cases.pickle", "rb") as f:
            self.cases = pickle.load(f)

        self.benchmarks = {}
        for key in self.cases:
            result_folder = self.folder / "results" / self.key_to_str(key)
            if result_folder.exists():
                result = self.benchmark_class.load_folder(result_folder)
                benchmark = self.create_benchmark(key)
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

            if keep and result_folder.exists():
                continue
            elif not keep and result_folder.exists():
                self.remove_benchmark(key)
            job_keys.append(key)

        for key in job_keys:
            benchmark = self.create_benchmark(key)
            t0 = time.perf_counter()
            benchmark.run()
            t1 = time.perf_counter()
            self.benchmarks[key] = benchmark
            bench_folder = self.folder / "results" / self.key_to_str(key)
            bench_folder.mkdir(exist_ok=True)
            benchmark.save_run(bench_folder)
            benchmark.result["run_time"] = float(t1 - t0)
            benchmark.save_main(bench_folder)

    def set_colors(self, colors=None, map_name="tab20"):
        if colors is None:
            case_keys = list(self.cases.keys())
            self.colors = get_some_colors(
                case_keys, map_name=map_name, color_engine="matplotlib", shuffle=False, margin=0
            )
        else:
            self.colors = colors

    def get_colors(self):
        if self.colors is None:
            self.set_colors()
        return self.colors

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

    def plot_run_times(self, case_keys=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())
        run_times = self.get_run_times(case_keys=case_keys)

        colors = self.get_colors()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        labels = []
        for i, key in enumerate(case_keys):
            labels.append(self.cases[key]["label"])
            rt = run_times.at[key, "run_times"]
            ax.bar(i, rt, width=0.8, color=colors[key])
        ax.set_xticks(np.arange(len(case_keys)))
        ax.set_xticklabels(labels, rotation=45.0)
        return fig

        # ax = run_times.plot(kind="bar")
        # return ax.figure

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

    def get_templates(self, key, operator="average"):
        sorting_analyzer = self.get_sorting_analyzer(case_key=key)
        templates = sorting_analyzer.get_extenson("templates").get_data(operator=operator)
        return templates

    def compute_metrics(self, case_keys=None, metric_names=["snr", "firing_rate"], force=False):
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
                qm_ext = sorting_analyzer.compute("quality_metrics", metric_names=metric_names)

            # TODO remove this metics CSV file!!!!
            metrics = qm_ext.get_data()
            # metrics.to_csv(filename, sep="\t", index=True)

    def get_metrics(self, key):
        import pandas as pd

        dataset_key = self.cases[key]["dataset"]

        analyzer = self.get_sorting_analyzer(key)
        ext = analyzer.get_extension("quality_metrics")
        if ext is None:
            # TODO au to compute ????
            return None

        metrics = ext.get_data()
        return metrics

        # filename = self.folder / "metrics" / f"{self.key_to_str(dataset_key)}.csv"
        # if not filename.exists():
        #     return
        # metrics = pd.read_csv(filename, sep="\t", index_col=0)
        # dataset_key = self.cases[key]["dataset"]
        # recording, gt_sorting = self.datasets[dataset_key]
        # metrics.index = gt_sorting.unit_ids
        # return metrics

    def get_units_snr(self, key):
        """ """
        return self.get_metrics(key)["snr"]

    def get_result(self, key):
        return self.benchmarks[key].result


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

                result[k] = load_extractor(folder / k)
            elif format == "Motion":
                from spikeinterface.sortingcomponents.motion import Motion

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


def _simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
