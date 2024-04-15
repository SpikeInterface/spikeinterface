from __future__ import annotations

from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd

import time

import os

from spikeinterface.core.core_tools import check_json
from spikeinterface import load_extractor, split_job_kwargs, create_sorting_analyzer, load_sorting_analyzer

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
        self.cases = {}
        self.benchmarks = {}
        self.scan_folder()

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

        (study_folder / "datasets").mkdir()
        (study_folder / "datasets" / "recordings").mkdir()
        (study_folder / "datasets" / "gt_sortings").mkdir()
        (study_folder / "run_logs").mkdir()
        (study_folder / "metrics").mkdir()
        (study_folder / "results").mkdir()

        for key, (rec, gt_sorting) in datasets.items():
            assert "/" not in key, "'/' cannot be in the key name!"
            assert "\\" not in key, "'\\' cannot be in the key name!"

            # recordings are pickled
            rec.dump_to_pickle(study_folder / f"datasets/recordings/{key}.pickle")

            # sortings are pickled + saved as NumpyFolderSorting
            gt_sorting.dump_to_pickle(study_folder / f"datasets/gt_sortings/{key}.pickle")
            gt_sorting.save(format="numpy_folder", folder=study_folder / f"datasets/gt_sortings/{key}")

        info = {}
        info["levels"] = levels
        (study_folder / "info.json").write_text(json.dumps(info, indent=4), encoding="utf8")

        # cases is dumped to a pickle file, json is not possible because of the tuple key
        (study_folder / "cases.pickle").write_bytes(pickle.dumps(cases))

        return cls(study_folder)

    def create_benchmark(self):
        raise NotImplementedError

    def scan_folder(self):
        if not (self.folder / "datasets").exists():
            raise ValueError(f"This is folder is not a GroundTruthStudy : {self.folder.absolute()}")

        with open(self.folder / "info.json", "r") as f:
            self.info = json.load(f)

        self.levels = self.info["levels"]

        for rec_file in (self.folder / "datasets" / "recordings").glob("*.pickle"):
            key = rec_file.stem
            rec = load_extractor(rec_file)
            gt_sorting = load_extractor(self.folder / f"datasets" / "gt_sortings" / key)
            self.datasets[key] = (rec, gt_sorting)

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

    def get_run_times(self, case_keys=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())

        run_times = {}
        for key in case_keys:
            benchmark = self.benchmarks[key]
            assert benchmark is not None
            run_times[key] = benchmark.result["run_time"]

        df = pd.DataFrame(dict(run_times=run_times))
        if not isinstance(self.levels, str):
            df.index.names = self.levels
        return df

    def plot_run_times(self, case_keys=None):
        if case_keys is None:
            case_keys = list(self.cases.keys())
        run_times = self.get_run_times(case_keys=case_keys)

        run_times.plot(kind="bar")

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
        if case_keys is None:
            case_keys = self.cases.keys()

        base_folder = self.folder / "sorting_analyzer"
        base_folder.mkdir(exist_ok=True)

        dataset_keys = [self.cases[key]["dataset"] for key in case_keys]
        dataset_keys = set(dataset_keys)
        for dataset_key in dataset_keys:
            # the waveforms depend on the dataset key
            folder = base_folder / self.key_to_str(dataset_key)
            recording, gt_sorting = self.datasets[dataset_key]
            sorting_analyzer = create_sorting_analyzer(
                gt_sorting, recording, format="binary_folder", folder=folder, return_scaled=return_scaled
            )
            sorting_analyzer.compute("random_spikes", **random_params)
            sorting_analyzer.compute("templates", **job_kwargs)
            sorting_analyzer.compute("noise_levels")

    def get_sorting_analyzer(self, case_key=None, dataset_key=None):
        if case_key is not None:
            dataset_key = self.cases[case_key]["dataset"]

        folder = self.folder / "sorting_analyzer" / self.key_to_str(dataset_key)
        sorting_analyzer = load_sorting_analyzer(folder)
        return sorting_analyzer

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
                # some case can share the same waveform extractor
                continue
            done.append(dataset_key)
            filename = self.folder / "metrics" / f"{self.key_to_str(dataset_key)}.csv"
            if filename.exists():
                if force:
                    os.remove(filename)
                else:
                    continue
            sorting_analyzer = self.get_sorting_analyzer(key)
            qm_ext = sorting_analyzer.compute("quality_metrics", metric_names=metric_names)
            metrics = qm_ext.get_data()
            metrics.to_csv(filename, sep="\t", index=True)

    def get_metrics(self, key):
        import pandas as pd

        dataset_key = self.cases[key]["dataset"]

        filename = self.folder / "metrics" / f"{self.key_to_str(dataset_key)}.csv"
        if not filename.exists():
            return
        metrics = pd.read_csv(filename, sep="\t", index_col=0)
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        metrics.index = gt_sorting.unit_ids
        return metrics

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
            if format == "npy":
                np.save(folder / f"{k}.npy", self.result[k])
            elif format == "pickle":
                with open(folder / f"{k}.pickle", mode="wb") as f:
                    pickle.dump(self.result[k], f)
            elif format == "sorting":
                self.result[k].save(folder=folder / k, format="numpy_folder", overwrite=True)
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
