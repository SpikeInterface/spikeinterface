from __future__ import annotations

from pathlib import Path
import shutil
import json
import numpy as np

import os

from spikeinterface.core.core_tools import check_json
from spikeinterface import load_extractor, split_job_kwargs, create_sorting_analyzer, load_sorting_analyzer

import pickle

_key_separator = "_-°°-_"

class BenchmarkStudy:
    """
    Manage a list of Benchmark
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
        for f in (log_file, ):
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
            benchmark.run()
            self.benchmarks[key] = benchmark
            bench_folder = self.folder / "results" / self.key_to_str(key)
            bench_folder.mkdir(exist_ok=True)
            benchmark.save_run(bench_folder)
    
    def compute_results(self, case_keys=None, verbose=False, **result_params):
        if case_keys is None:
            case_keys = self.cases.keys()

        job_keys = []
        for key in case_keys:
            benchmark = self.benchmarks[key]
            assert benchmark is not None
            benchmark.compute_result(**result_params)
            benchmark.save_result(self.folder / "results" / self.key_to_str(key))

    def create_sorting_analyzer_gt(self, case_keys=None, **kwargs):
        if case_keys is None:
            case_keys = self.cases.keys()

        select_params, job_kwargs = split_job_kwargs(kwargs)

        base_folder = self.folder / "sorting_analyzer"
        base_folder.mkdir(exist_ok=True)

        dataset_keys = [self.cases[key]["dataset"] for key in case_keys]
        dataset_keys = set(dataset_keys)
        for dataset_key in dataset_keys:
            # the waveforms depend on the dataset key
            folder = base_folder / self.key_to_str(dataset_key)
            recording, gt_sorting = self.datasets[dataset_key]
            sorting_analyzer = create_sorting_analyzer(gt_sorting, recording, format="binary_folder", folder=folder)
            sorting_analyzer.select_random_spikes(**select_params)
            sorting_analyzer.compute("waveforms", **job_kwargs)
            sorting_analyzer.compute("templates")
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
    """
    def __init__(self):
        self.result = {}

    _run_key_saved = []
    _result_key_saved = []

    def _save_keys(self, saved_keys, folder):
        for k, format in saved_keys:
            if format == "npy":
                np.save(folder / f"{k}.npy", self.result[k])
            elif format =="pickle":
                with open(folder  / f"{k}.pickle", mode="wb") as f:
                    pickle.dump(self.result[k], f)
            elif format == 'sorting':
                self.result[k].save(folder = folder / k, format="numpy_folder")
            elif format == 'zarr_templates':
                self.result[k].to_zarr(folder / k)
            elif format == 'sorting_analyzer':
                pass
            else:
                raise ValueError(f"Save error {k} {format}")

    def save_run(self, folder):
        self._save_keys(self._run_key_saved, folder)
    
    def save_result(self, folder):
        self._save_keys(self._result_key_saved, folder)

    @classmethod
    def load_folder(cls, folder):
        result = {}
        for k, format in cls._run_key_saved + cls._result_key_saved:
            if format == "npy":
                file = folder / f"{k}.npy"
                if file.exists():
                    result[k] = np.load(file)
            elif format =="pickle":
                file = folder / f"{k}.pickle"
                if file.exists():
                    with open(file, mode="rb") as f:
                        result[k] = pickle.load(f)
            elif format =="sorting":
                from spikeinterface.core import load_extractor
                result[k] = load_extractor(folder / k)
            elif format =="zarr_templates":
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


# class BenchmarkBaseOld:
#     _array_names = ()
#     _waveform_names = ()
#     _sorting_names = ()

#     _array_names_from_parent = ()
#     _waveform_names_from_parent = ()
#     _sorting_names_from_parent = ()

#     def __init__(
#         self,
#         folder=None,
#         title="",
#         overwrite=None,
#         job_kwargs={"chunk_duration": "1s", "n_jobs": -1, "progress_bar": True, "verbose": True},
#         parent_benchmark=None,
#     ):
#         self.folder = Path(folder)
#         self.title = title
#         self.overwrite = overwrite
#         self.job_kwargs = job_kwargs
#         self.run_times = None

#         self._args = []
#         self._kwargs = dict(title=title, overwrite=overwrite, job_kwargs=job_kwargs)

#         self.waveforms = {}
#         self.sortings = {}

#         self.parent_benchmark = parent_benchmark

#         if self.parent_benchmark is not None:
#             for name in self._array_names_from_parent:
#                 setattr(self, name, getattr(parent_benchmark, name))

#             for name in self._waveform_names_from_parent:
#                 self.waveforms[name] = parent_benchmark.waveforms[name]

#             for key in parent_benchmark.sortings.keys():
#                 if isinstance(key, str) and key in self._sorting_names_from_parent:
#                     self.sortings[key] = parent_benchmark.sortings[key]
#                 elif isinstance(key, tuple) and key[0] in self._sorting_names_from_parent:
#                     self.sortings[key] = parent_benchmark.sortings[key]

#     def save_to_folder(self):
#         if self.folder.exists():
#             import glob, os

#             pattern = "*.*"
#             files = self.folder.glob(pattern)
#             for file in files:
#                 if file.is_file():
#                     os.remove(file)
#         else:
#             self.folder.mkdir(parents=True)

#         if self.parent_benchmark is None:
#             parent_folder = None
#         else:
#             parent_folder = str(self.parent_benchmark.folder)

#         info = {
#             "args": self._args,
#             "kwargs": self._kwargs,
#             "parent_folder": parent_folder,
#         }
#         info = check_json(info)
#         (self.folder / "info.json").write_text(json.dumps(info, indent=4), encoding="utf8")

#         for name in self._array_names:
#             if self.parent_benchmark is not None and name in self._array_names_from_parent:
#                 continue
#             value = getattr(self, name)
#             if value is not None:
#                 np.save(self.folder / f"{name}.npy", value)

#         if self.run_times is not None:
#             run_times_filename = self.folder / "run_times.json"
#             run_times_filename.write_text(json.dumps(self.run_times, indent=4), encoding="utf8")

#         for key, sorting in self.sortings.items():
#             (self.folder / "sortings").mkdir(exist_ok=True)
#             if isinstance(key, str):
#                 npz_file = self.folder / "sortings" / (str(key) + ".npz")
#             elif isinstance(key, tuple):
#                 npz_file = self.folder / "sortings" / ("_###_".join(key) + ".npz")
#             NpzSortingExtractor.write_sorting(sorting, npz_file)

#     @classmethod
#     def load_from_folder(cls, folder, parent_benchmark=None):
#         folder = Path(folder)
#         assert folder.exists()

#         with open(folder / "info.json", "r") as f:
#             info = json.load(f)
#         args = info["args"]
#         kwargs = info["kwargs"]

#         if info["parent_folder"] is None:
#             parent_benchmark = None
#         else:
#             if parent_benchmark is None:
#                 parent_benchmark = cls.load_from_folder(info["parent_folder"])

#         import os

#         kwargs["folder"] = folder

#         bench = cls(*args, **kwargs, parent_benchmark=parent_benchmark)

#         for name in cls._array_names:
#             filename = folder / f"{name}.npy"
#             if filename.exists():
#                 arr = np.load(filename)
#             else:
#                 arr = None
#             setattr(bench, name, arr)

#         if (folder / "run_times.json").exists():
#             with open(folder / "run_times.json", "r") as f:
#                 bench.run_times = json.load(f)
#         else:
#             bench.run_times = None

#         for key in bench._waveform_names:
#             if parent_benchmark is not None and key in bench._waveform_names_from_parent:
#                 continue
#             waveforms_folder = folder / "waveforms" / key
#             if waveforms_folder.exists():
#                 bench.waveforms[key] = load_waveforms(waveforms_folder, with_recording=True)

#         sorting_folder = folder / "sortings"
#         if sorting_folder.exists():
#             for npz_file in sorting_folder.glob("*.npz"):
#                 name = npz_file.stem
#                 if "_###_" in name:
#                     key = tuple(name.split("_###_"))
#                 else:
#                     key = name
#                 bench.sortings[key] = NpzSortingExtractor(npz_file)

#         return bench
