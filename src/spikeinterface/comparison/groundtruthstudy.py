from pathlib import Path
import shutil
import os
import json
import pickle

import numpy as np

from spikeinterface.core import load_extractor, extract_waveforms, load_waveforms
from spikeinterface.core.core_tools import SIJsonEncoder

from spikeinterface.sorters import run_sorter_jobs, read_sorter_folder

from spikeinterface import WaveformExtractor
from spikeinterface.qualitymetrics import compute_quality_metrics

from .paircomparisons import compare_sorter_to_ground_truth, GroundTruthComparison


# TODO : save comparison in folders when COmparison object will be able to serialize
# TODO ??: make an internal optional binary copy when running several external sorter
# on the same dataset to avoid multiple save binary ? even when the recording is float32 (ks need int16)



# This is to separate names when the key are tuples when saving folders
_key_separator = " ## "
# This would be more funny
# _key_separator = " (°_°) "


class GroundTruthStudy:
    """
    This class is an helper function to run any comparison on several "cases" for several ground truth dataset.

    "cases" can be:
      * several sorter for comparisons
      * same sorter with differents parameters
      * parameters of comparisons
      * any combination of theses
    
    For enough flexibility cases key can be a tuple so that we can varify complexity along several
    "levels" or "axis" (paremeters or sorter).

    Generated dataframes will have index with several levels optionaly.
    
    Ground truth dataset need recording+sorting. This can be from mearec file or from the internal generator
    :py:fun:`generate_ground_truth_recording()`
    
    This GroundTruthStudy have been refactor in version 0.100 to be more flexible than previous versions.
    Folders structures are not backward compatible at all.
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
            assert all(len(key) == num_levels for key in cases.keys()), "Keys for cases are not homogeneous, tuple negth differ"
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
        (study_folder / "datasets/recordings").mkdir()
        (study_folder / "datasets/gt_sortings").mkdir()
        (study_folder / "sorters").mkdir()
        (study_folder / "sortings").mkdir()
        (study_folder / "sortings" / "run_logs").mkdir()
        (study_folder / "metrics").mkdir()

        for key, (rec, gt_sorting) in datasets.items():
            assert "/" not in key
            assert "\\" not in key

            # rec are pickle
            rec.dump_to_pickle(study_folder / f"datasets/recordings/{key}.pickle")

            # sorting are pickle + saved as NumpyFolderSorting
            gt_sorting.dump_to_pickle(study_folder / f"datasets/gt_sortings/{key}.pickle")
            gt_sorting.save(format="numpy_folder", folder=study_folder / f"datasets/gt_sortings/{key}")
        
        
        info = {}
        info["levels"] = levels
        (study_folder / "info.json").write_text(json.dumps(info, indent=4), encoding="utf8")

        # (study_folder / "cases.jon").write_text(
        #     json.dumps(cases, indent=4, cls=SIJsonEncoder),
        #     encoding="utf8",
        # )
        # cases is dump to a pickle file, json is not possible because of tuple key
        (study_folder / "cases.pickle").write_bytes(pickle.dumps(cases))

        return cls(study_folder)


    def scan_folder(self):
        if not (self.folder / "datasets").exists():
            raise ValueError(f"This is folder is not a GroundTruthStudy : {self.folder.absolute()}")

        with open(self.folder / "info.json", "r") as f:
            self.info = json.load(f)
        
        self.levels = self.info["levels"]
        # if isinstance(self.levels, list):
        #     # because tuple caoont be stored in json
        #     self.levels = tuple(self.info["levels"])

        for rec_file in (self.folder / "datasets/recordings").glob("*.pickle"):
            key = rec_file.stem
            rec = load_extractor(rec_file)
            gt_sorting = load_extractor(self.folder / f"datasets/gt_sortings/{key}")
            self.datasets[key] = (rec, gt_sorting)
        
        with open(self.folder / "cases.pickle", "rb") as f:
            self.cases = pickle.load(f)

        self.comparisons = {k: None for k in self.cases}

        self.sortings = {}
        for key in self.cases:
            sorting_folder = self.folder / "sortings" / self.key_to_str(key)
            if sorting_folder.exists():
                sorting = load_extractor(sorting_folder)
            else:
                sorting = None
            self.sortings[key] = sorting


    def __repr__(self):
        t = f"GroundTruthStudy {self.folder.stem} \n"
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

    def run_sorters(self, case_keys=None, engine='loop', engine_kwargs={}, keep=True, verbose=False):
        """
        
        """
        if case_keys is None:
            case_keys = self.cases.keys()

        job_list = []
        for key in case_keys:
            sorting_folder = self.folder / "sortings" / self.key_to_str(key)
            sorting_exists = sorting_folder.exists()

            sorter_folder = self.folder / "sorters" / self.key_to_str(key)
            sorter_folder_exists = sorting_folder.exists()

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

            if sorting_exists:
                # TODO : delete sorting + log
                pass

            params = self.cases[key]["run_sorter_params"].copy()
            # this ensure that sorter_name is given
            recording, _ = self.datasets[self.cases[key]["dataset"]]
            sorter_name = params.pop("sorter_name")
            job = dict(sorter_name=sorter_name,
                       recording=recording,
                       output_folder=sorter_folder)
            job.update(params)
            # the verbose is overwritten and global to all run_sorters
            job["verbose"] = verbose
            job_list.append(job)

        run_sorter_jobs(job_list, engine=engine, engine_kwargs=engine_kwargs, return_output=False)

        # TODO create a list in laucher for engine blocking and non-blocking
        if engine not in ("slurm", ):
            self.copy_sortings(case_keys)

    def copy_sortings(self, case_keys=None, force=True):
        if case_keys is None:
            case_keys = self.cases.keys()
        
        for key in case_keys:
            sorting_folder = self.folder / "sortings" / self.key_to_str(key)
            sorter_folder = self.folder / "sorters" / self.key_to_str(key)
            log_file = self.folder / "sortings" / "run_logs" / f"{self.key_to_str(key)}.json"

            sorting = read_sorter_folder(sorter_folder, raise_error=False)
            if sorting is not None:
                if sorting_folder.exists():
                    if force:
                        # TODO delete folder + log
                        shutil.rmtree(sorting_folder)
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

    def extract_waveforms_gt(self, case_keys=None, **extract_kwargs):

        if case_keys is None:
            case_keys = self.cases.keys()

        base_folder = self.folder / "waveforms"
        base_folder.mkdir(exist_ok=True)

        for key in case_keys:
            dataset_key = self.cases[key]["dataset"]
            recording, gt_sorting = self.datasets[dataset_key]
            wf_folder = base_folder / self.key_to_str(key)
            we = extract_waveforms(recording, gt_sorting, folder=wf_folder)

    def get_waveform_extractor(self, key):
        # some recording are not dumpable to json and the waveforms extactor need it!
        # so we load it with and put after
        we = load_waveforms(self.folder / "waveforms" / self.key_to_str(key), with_recording=False)
        dataset_key = self.cases[key]["dataset"]
        recording, _ = self.datasets[dataset_key]        
        we.set_recording(recording)
        return we

    def get_templates(self, key, mode="mean"):
        we = self.get_waveform_extractor(key)
        templates = we.get_all_templates(mode=mode)
        return templates

    def compute_metrics(self, case_keys=None, metric_names=["snr", "firing_rate"], force=False):
        if case_keys is None:
            case_keys = self.cases.keys()
        
        for key in case_keys:
            filename = self.folder / "metrics" / f"{self.key_to_str(key)}.txt"
            if filename.exists():
                if force:
                    os.remove(filename)
                else:
                    continue

            we = self.get_waveform_extractor(key)
            metrics = compute_quality_metrics(we, metric_names=metric_names)
            metrics.to_csv(filename, sep="\t", index=True)

    def get_metrics(self, key):
        import pandas  as pd
        filename = self.folder / "metrics" / f"{self.key_to_str(key)}.txt"
        if not filename.exists():
            return
        metrics = pd.read_csv(filename, sep="\t", index_col=0)
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]        
        metrics.index = gt_sorting.unit_ids
        return metrics

    def get_units_snr(self, key):
        """
        """
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

    def get_count_units(
            self, case_keys=None, well_detected_score=None, redundant_score=None, overmerged_score=None
        ):

        import pandas as pd

        if case_keys is None:
            case_keys = list(self.cases.keys())

        if isinstance(case_keys[0], str):
            index = pd.Index(case_keys, name=self.levels)
        else:
            index = pd.MultiIndex.from_tuples(case_keys, names=self.levels)


        columns = ["num_gt", "num_sorter", "num_well_detected", "num_redundant", "num_overmerged"]
        comp = self.comparisons[case_keys[0]]
        if comp.exhaustive_gt:
            columns.extend(["num_false_positive", "num_bad"])
        count_units = pd.DataFrame(index=index, columns=columns, dtype=int)


        for key in case_keys:
            comp = self.comparisons.get(key, None)
            assert comp is not None, "You need to do study.run_comparisons() first"

            gt_sorting = comp.sorting1
            sorting = comp.sorting2

            count_units.loc[key, "num_gt"] = len(gt_sorting.get_unit_ids())
            count_units.loc[key, "num_sorter"] = len(sorting.get_unit_ids())
            count_units.loc[key, "num_well_detected"] = comp.count_well_detected_units(
                well_detected_score
            )
            if comp.exhaustive_gt:
                count_units.loc[key, "num_overmerged"] = comp.count_overmerged_units(
                    overmerged_score
                )
                count_units.loc[key, "num_redundant"] = comp.count_redundant_units(redundant_score)
                count_units.loc[key, "num_false_positive"] = comp.count_false_positive_units(
                    redundant_score
                )
                count_units.loc[key, "num_bad"] = comp.count_bad_units()

        return count_units

