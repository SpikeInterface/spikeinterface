from pathlib import Path
import shutil

import numpy as np

from spikeinterface.core import load_extractor
from spikeinterface.extractors import NpzSortingExtractor
from spikeinterface.sorters import sorter_dict, run_sorters

from spikeinterface import WaveformExtractor
from spikeinterface.qualitymetrics import compute_quality_metrics

from .paircomparisons import compare_sorter_to_ground_truth

from .studytools import (
    setup_comparison_study,
    get_rec_names,
    get_recordings,
    iter_working_folder,
    iter_computed_names,
    iter_computed_sorting,
    collect_run_times,
)


class GroundTruthStudy:
    def __init__(self, study_folder=None):
        import pandas as pd

        self.study_folder = Path(study_folder)
        self._is_scanned = False
        self.computed_names = None
        self.rec_names = None
        self.sorter_names = None

        self.scan_folder()

        self.comparisons = None
        self.exhaustive_gt = None

    def __repr__(self):
        t = "Ground truth study\n"
        t += "  " + str(self.study_folder) + "\n"
        t += "  recordings: {} {}\n".format(len(self.rec_names), self.rec_names)
        if len(self.sorter_names):
            t += "  sorters: {} {}\n".format(len(self.sorter_names), self.sorter_names)

        return t

    def scan_folder(self):
        self.rec_names = get_rec_names(self.study_folder)
        # scan computed names
        self.computed_names = list(iter_computed_names(self.study_folder))  # list of pair (rec_name, sorter_name)
        self.sorter_names = np.unique([e for _, e in iter_computed_names(self.study_folder)]).tolist()
        self._is_scanned = True

    @classmethod
    def create(cls, study_folder, gt_dict, **job_kwargs):
        setup_comparison_study(study_folder, gt_dict, **job_kwargs)
        return cls(study_folder)

    def run_sorters(self, sorter_list, mode_if_folder_exists="keep", remove_sorter_folders=False, **kwargs):
        sorter_folders = self.study_folder / "sorter_folders"
        recording_dict = get_recordings(self.study_folder)

        run_sorters(
            sorter_list,
            recording_dict,
            sorter_folders,
            with_output=False,
            mode_if_folder_exists=mode_if_folder_exists,
            **kwargs,
        )

        # results are copied so the heavy sorter_folders can be removed
        self.copy_sortings()

        if remove_sorter_folders:
            shutil.rmtree(self.study_folder / "sorter_folders")

    def _check_rec_name(self, rec_name):
        if not self._is_scanned:
            self.scan_folder()
        if len(self.rec_names) > 1 and rec_name is None:
            raise Exception("Pass 'rec_name' parameter to select which recording to use.")
        elif len(self.rec_names) == 1:
            rec_name = self.rec_names[0]
        else:
            rec_name = self.rec_names[self.rec_names.index(rec_name)]
        return rec_name

    def get_ground_truth(self, rec_name=None):
        rec_name = self._check_rec_name(rec_name)
        sorting = load_extractor(self.study_folder / "ground_truth" / rec_name)
        return sorting

    def get_recording(self, rec_name=None):
        rec_name = self._check_rec_name(rec_name)
        rec = load_extractor(self.study_folder / "raw_files" / rec_name)
        return rec

    def get_sorting(self, sort_name, rec_name=None):
        rec_name = self._check_rec_name(rec_name)

        selected_sorting = None
        if sort_name in self.sorter_names:
            for r_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
                if sort_name == sorter_name and r_name == rec_name:
                    selected_sorting = sorting
        return selected_sorting

    def copy_sortings(self):
        sorter_folders = self.study_folder / "sorter_folders"
        sorting_folders = self.study_folder / "sortings"
        log_olders = self.study_folder / "sortings" / "run_log"

        log_olders.mkdir(parents=True, exist_ok=True)

        for rec_name, sorter_name, output_folder in iter_working_folder(sorter_folders):
            SorterClass = sorter_dict[sorter_name]
            fname = rec_name + "[#]" + sorter_name
            npz_filename = sorting_folders / (fname + ".npz")

            try:
                sorting = SorterClass.get_result_from_folder(output_folder)
                NpzSortingExtractor.write_sorting(sorting, npz_filename)
            except:
                if npz_filename.is_file():
                    npz_filename.unlink()
            if (output_folder / "spikeinterface_log.json").is_file():
                shutil.copyfile(
                    output_folder / "spikeinterface_log.json", sorting_folders / "run_log" / (fname + ".json")
                )

        self.scan_folder()

    def run_comparisons(self, exhaustive_gt=False, **kwargs):
        self.comparisons = {}
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            sc = compare_sorter_to_ground_truth(gt_sorting, sorting, exhaustive_gt=exhaustive_gt, **kwargs)
            self.comparisons[(rec_name, sorter_name)] = sc
        self.exhaustive_gt = exhaustive_gt

    def aggregate_run_times(self):
        return collect_run_times(self.study_folder)

    def aggregate_performance_by_unit(self):
        assert self.comparisons is not None, "run_comparisons first"

        perf_by_unit = []
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            comp = self.comparisons[(rec_name, sorter_name)]

            perf = comp.get_performance(method="by_unit", output="pandas")
            perf["rec_name"] = rec_name
            perf["sorter_name"] = sorter_name
            perf = perf.reset_index()
            perf_by_unit.append(perf)

        import pandas as pd

        perf_by_unit = pd.concat(perf_by_unit)
        perf_by_unit = perf_by_unit.set_index(["rec_name", "sorter_name", "gt_unit_id"])

        return perf_by_unit

    def aggregate_count_units(self, well_detected_score=None, redundant_score=None, overmerged_score=None):
        assert self.comparisons is not None, "run_comparisons first"

        import pandas as pd

        index = pd.MultiIndex.from_tuples(self.computed_names, names=["rec_name", "sorter_name"])

        count_units = pd.DataFrame(
            index=index,
            columns=["num_gt", "num_sorter", "num_well_detected", "num_redundant", "num_overmerged"],
            dtype=int,
        )

        if self.exhaustive_gt:
            count_units["num_false_positive"] = pd.Series(dtype=int)
            count_units["num_bad"] = pd.Series(dtype=int)

        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = self.comparisons[(rec_name, sorter_name)]

            count_units.loc[(rec_name, sorter_name), "num_gt"] = len(gt_sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), "num_sorter"] = len(sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), "num_well_detected"] = comp.count_well_detected_units(
                well_detected_score
            )
            if self.exhaustive_gt:
                count_units.loc[(rec_name, sorter_name), "num_overmerged"] = comp.count_overmerged_units(
                    overmerged_score
                )
                count_units.loc[(rec_name, sorter_name), "num_redundant"] = comp.count_redundant_units(redundant_score)
                count_units.loc[(rec_name, sorter_name), "num_false_positive"] = comp.count_false_positive_units(
                    redundant_score
                )
                count_units.loc[(rec_name, sorter_name), "num_bad"] = comp.count_bad_units()

        return count_units

    def aggregate_dataframes(self, copy_into_folder=True, **karg_thresh):
        dataframes = {}
        dataframes["run_times"] = self.aggregate_run_times().reset_index()
        perfs = self.aggregate_performance_by_unit()

        dataframes["perf_by_unit"] = perfs.reset_index()
        dataframes["count_units"] = self.aggregate_count_units(**karg_thresh).reset_index()

        if copy_into_folder:
            tables_folder = self.study_folder / "tables"
            tables_folder.mkdir(parents=True, exist_ok=True)

            for name, df in dataframes.items():
                df.to_csv(str(tables_folder / (name + ".csv")), sep="\t", index=False)

        return dataframes

    def get_waveform_extractor(self, rec_name, sorter_name=None):
        rec = self.get_recording(rec_name)

        if sorter_name is None:
            name = "GroundTruth"
            sorting = self.get_ground_truth(rec_name)
        else:
            assert sorter_name in self.sorter_names
            name = sorter_name
            sorting = self.get_sorting(sorter_name, rec_name)

        waveform_folder = self.study_folder / "waveforms" / f"waveforms_{name}_{rec_name}"

        if waveform_folder.is_dir():
            we = WaveformExtractor.load(waveform_folder)
        else:
            we = WaveformExtractor.create(rec, sorting, waveform_folder)
        return we

    def compute_waveforms(
        self,
        rec_name,
        sorter_name=None,
        ms_before=3.0,
        ms_after=4.0,
        max_spikes_per_unit=500,
        n_jobs=-1,
        total_memory="1G",
    ):
        we = self.get_waveform_extractor(rec_name, sorter_name)
        we.set_params(ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=max_spikes_per_unit)
        we.run_extract_waveforms(n_jobs=n_jobs, total_memory=total_memory)

    def get_templates(self, rec_name, sorter_name=None, mode="median"):
        """
        Get template for a given recording.

        If sorter_name=None then template are from the ground truth.

        """
        we = self.get_waveform_extractor(rec_name, sorter_name=sorter_name)
        templates = we.get_all_templates(mode=mode)
        return templates

    def compute_metrics(
        self,
        rec_name,
        metric_names=["snr"],
        ms_before=3.0,
        ms_after=4.0,
        max_spikes_per_unit=500,
        n_jobs=-1,
        total_memory="1G",
    ):
        we = self.get_waveform_extractor(rec_name)
        we.set_params(ms_before=ms_before, ms_after=ms_after, max_spikes_per_unit=max_spikes_per_unit)
        we.run_extract_waveforms(n_jobs=n_jobs, total_memory=total_memory)

        # metrics
        metrics = compute_quality_metrics(we, metric_names=metric_names)
        folder = self.study_folder / "metrics"
        folder.mkdir(exist_ok=True)
        filename = folder / f"metrics _{rec_name}.txt"
        metrics.to_csv(filename, sep="\t", index=True)

        return metrics

    def get_metrics(self, rec_name=None, **metric_kwargs):
        """
        Load or compute units metrics  for a given recording.
        """
        rec_name = self._check_rec_name(rec_name)
        metrics_folder = self.study_folder / "metrics"
        metrics_folder.mkdir(parents=True, exist_ok=True)

        filename = self.study_folder / "metrics" / f"metrics _{rec_name}.txt"
        import pandas as pd

        if filename.is_file():
            metrics = pd.read_csv(filename, sep="\t", index_col=0)
            gt_sorting = self.get_ground_truth(rec_name)
            metrics.index = gt_sorting.unit_ids
        else:
            metrics = self.compute_metrics(rec_name, **metric_kwargs)

        metrics.index.name = "unit_id"
        #  add rec name columns
        metrics["rec_name"] = rec_name

        return metrics

    def get_units_snr(self, rec_name=None, **metric_kwargs):
        """ """
        metric = self.get_metrics(rec_name=rec_name, **metric_kwargs)
        return metric["snr"]

    def concat_all_snr(self):
        metrics = []
        for rec_name in self.rec_names:
            df = self.get_metrics(rec_name)
            df = df.reset_index()
            metrics.append(df)
        metrics = pd.concat(metrics)
        metrics = metrics.set_index(["rec_name", "unit_id"])
        return metrics["snr"]
