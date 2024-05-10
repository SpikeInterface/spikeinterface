from __future__ import annotations

from pathlib import Path
import shutil
import os
import json
import pickle

import numpy as np

from spikeinterface.core import load_extractor, create_sorting_analyzer, load_sorting_analyzer
from spikeinterface.core.core_tools import SIJsonEncoder
from spikeinterface.core.job_tools import split_job_kwargs

from spikeinterface.sorters import run_sorter_jobs, read_sorter_folder

from spikeinterface.qualitymetrics import compute_quality_metrics

from .paircomparisons import compare_sorter_to_ground_truth, GroundTruthComparison


# TODO later : save comparison in folders when comparison object will be able to serialize


# This is to separate names when the key are tuples when saving folders
# _key_separator = "_##_"
_key_separator = "_-°°-_"


class GroundTruthStudy:
    """
    This class is an helper function to run any comparison on several "cases" for many ground-truth dataset.

    "cases" refer to:
      * several sorters for comparisons
      * same sorter with differents parameters
      * any combination of these (and more)

    For increased flexibility, cases keys can be a tuple so that we can vary complexity along several
    "levels" or "axis" (paremeters or sorters).
    In this case, the result dataframes will have `MultiIndex` to handle the different levels.

    A ground-truth dataset is made of a `Recording` and a `Sorting` object. For example, it can be a simulated dataset with MEArec or internally generated (see
    :py:func:`~spikeinterface.core.generate.generate_ground_truth_recording()`).

    This GroundTruthStudy have been refactor in version 0.100 to be more flexible than previous versions.
    Note that the underlying folder structure is not backward compatible!
    """

    def __init__(self, study_folder):
        self.folder = Path(study_folder)

        self.datasets = {}
        self.cases = {}
        self.sortings = {}
        self.comparisons = {}

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
        (study_folder / "sorters").mkdir()
        (study_folder / "sortings").mkdir()
        (study_folder / "sortings" / "run_logs").mkdir()
        (study_folder / "metrics").mkdir()
        (study_folder / "comparisons").mkdir()

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

        self.sortings = {k: None for k in self.cases}
        self.comparisons = {k: None for k in self.cases}
        for key in self.cases:
            sorting_folder = self.folder / "sortings" / self.key_to_str(key)
            if sorting_folder.exists():
                self.sortings[key] = load_extractor(sorting_folder)

            comparison_file = self.folder / "comparisons" / (self.key_to_str(key) + ".pickle")
            if comparison_file.exists():
                with open(comparison_file, mode="rb") as f:
                    try:
                        self.comparisons[key] = pickle.load(f)
                    except Exception:
                        pass

    def __repr__(self):
        t = f"{self.__class__.__name__} {self.folder.stem} \n"
        t += f"  datasets: {len(self.datasets)} {list(self.datasets.keys())}\n"
        t += f"  cases: {len(self.cases)} {list(self.cases.keys())}\n"
        num_computed = sum([1 for sorting in self.sortings.values() if sorting is not None])
        t += f"  computed: {num_computed}\n"

        return t

    def key_to_str(self, key):
        if isinstance(key, str):
            return key
        elif isinstance(key, tuple):
            return _key_separator.join(key)
        else:
            raise ValueError("Keys for cases must str or tuple")

    def remove_sorting(self, key):
        sorting_folder = self.folder / "sortings" / self.key_to_str(key)
        log_file = self.folder / "sortings" / "run_logs" / f"{self.key_to_str(key)}.json"
        comparison_file = self.folder / "comparisons" / self.key_to_str(key)
        if sorting_folder.exists():
            shutil.rmtree(sorting_folder)
        for f in (log_file, comparison_file):
            if f.exists():
                f.unlink()

    def run_sorters(self, case_keys=None, engine="loop", engine_kwargs={}, keep=True, verbose=False):
        if case_keys is None:
            case_keys = self.cases.keys()

        job_list = []
        for key in case_keys:
            sorting_folder = self.folder / "sortings" / self.key_to_str(key)
            sorting_exists = sorting_folder.exists()

            sorter_folder = self.folder / "sorters" / self.key_to_str(key)
            sorter_folder_exists = sorter_folder.exists()

            if keep:
                if sorting_exists:
                    continue
                if sorter_folder_exists:
                    # the sorter folder exists but havent been copied to sortings folder
                    sorting = read_sorter_folder(sorter_folder, raise_error=False)
                    if sorting is not None:
                        # save and skip
                        self.copy_sortings(case_keys=[key])
                        continue

            self.remove_sorting(key)

            if sorter_folder_exists:
                shutil.rmtree(sorter_folder)

            params = self.cases[key]["run_sorter_params"].copy()
            # this ensure that sorter_name is given
            recording, _ = self.datasets[self.cases[key]["dataset"]]
            sorter_name = params.pop("sorter_name")
            job = dict(
                sorter_name=sorter_name,
                recording=recording,
                output_folder=sorter_folder,
            )
            job.update(params)
            # the verbose is overwritten and global to all run_sorters
            job["verbose"] = verbose
            job["with_output"] = False
            job_list.append(job)

        run_sorter_jobs(job_list, engine=engine, engine_kwargs=engine_kwargs, return_output=False)

        # TODO later create a list in laucher for engine blocking and non-blocking
        if engine not in ("slurm",):
            self.copy_sortings(case_keys)

    def copy_sortings(self, case_keys=None, force=True):
        if case_keys is None:
            case_keys = self.cases.keys()

        for key in case_keys:
            sorting_folder = self.folder / "sortings" / self.key_to_str(key)
            sorter_folder = self.folder / "sorters" / self.key_to_str(key)
            log_file = self.folder / "sortings" / "run_logs" / f"{self.key_to_str(key)}.json"

            if (sorter_folder / "spikeinterface_log.json").exists():
                sorting = read_sorter_folder(
                    sorter_folder, raise_error=False, register_recording=False, sorting_info=False
                )
            else:
                sorting = None

            if sorting is not None:
                if sorting_folder.exists():
                    if force:
                        self.remove_sorting(key)
                    else:
                        continue

                sorting = sorting.save(format="numpy_folder", folder=sorting_folder)
                self.sortings[key] = sorting

                # copy logs
                shutil.copyfile(sorter_folder / "spikeinterface_log.json", log_file)

    def run_comparisons(self, case_keys=None, comparison_class=GroundTruthComparison, **kwargs):
        if case_keys is None:
            case_keys = self.cases.keys()

        for key in case_keys:
            dataset_key = self.cases[key]["dataset"]
            _, gt_sorting = self.datasets[dataset_key]
            sorting = self.sortings[key]
            if sorting is None:
                self.comparisons[key] = None
                continue
            comp = comparison_class(gt_sorting, sorting, **kwargs)
            self.comparisons[key] = comp

            comparison_file = self.folder / "comparisons" / (self.key_to_str(key) + ".pickle")
            with open(comparison_file, mode="wb") as f:
                pickle.dump(comp, f)

    def get_run_times(self, case_keys=None):
        import pandas as pd

        if case_keys is None:
            case_keys = self.cases.keys()

        log_folder = self.folder / "sortings" / "run_logs"

        run_times = {}
        for key in case_keys:
            log_file = log_folder / f"{self.key_to_str(key)}.json"
            with open(log_file, mode="r") as logfile:
                log = json.load(logfile)
                run_time = log.get("run_time", None)
            run_times[key] = run_time

        return pd.Series(run_times, name="run_time")

    def create_sorting_analyzer_gt(self, case_keys=None, random_params={}, waveforms_params={}, **job_kwargs):
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
            sorting_analyzer = create_sorting_analyzer(gt_sorting, recording, format="binary_folder", folder=folder)
            sorting_analyzer.compute("random_spikes", **random_params)
            sorting_analyzer.compute("templates", **job_kwargs)
            sorting_analyzer.compute("noise_levels")

    def get_sorting_analyzer(self, case_key=None, dataset_key=None):
        if case_key is not None:
            dataset_key = self.cases[case_key]["dataset"]

        folder = self.folder / "sorting_analyzer" / self.key_to_str(dataset_key)
        sorting_analyzer = load_sorting_analyzer(folder)
        return sorting_analyzer

    # def get_templates(self, key, mode="average"):
    #     analyzer = self.get_sorting_analyzer(case_key=key)
    #     templates = sorting_analyzer.get_all_templates(mode=mode)
    #     return templates

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
            analyzer = self.get_sorting_analyzer(key)
            metrics = compute_quality_metrics(analyzer, metric_names=metric_names)
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

    def get_performance_by_unit(self, case_keys=None):
        import pandas as pd

        if case_keys is None:
            case_keys = self.cases.keys()

        perf_by_unit = []
        for key in case_keys:
            comp = self.comparisons.get(key, None)
            assert comp is not None, "You need to do study.run_comparisons() first"

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
        comp = self.comparisons[case_keys[0]]
        if comp.exhaustive_gt:
            columns.extend(["num_false_positive", "num_redundant", "num_overmerged", "num_bad"])
        count_units = pd.DataFrame(index=index, columns=columns, dtype=int)

        for key in case_keys:
            comp = self.comparisons.get(key, None)
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
