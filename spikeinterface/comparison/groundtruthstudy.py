from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd

from spikeinterface.core import load_extractor
from spikeinterface.extractors import NpzSortingExtractor
from spikeinterface.sorters import sorter_dict, run_sorters



from .comparisontools import _perf_keys
from .groundtruthcomparison import compare_sorter_to_ground_truth

from .studytools import (setup_comparison_study,  get_rec_names, get_recordings,
        iter_output_folders, iter_computed_names, iter_computed_sorting, collect_run_times)

# import spikeinterface.toolkit as st


class GroundTruthStudy:
    def __init__(self, study_folder=None):
        self.study_folder = Path(study_folder)
        self._is_scanned = False
        self.computed_names = None
        self.rec_names = None
        self.sorter_names = None

        self.scan_folder()

        self.comparisons = None
        self.exhaustive_gt = None

    def __repr__(self):
        t = 'Groud truth study\n'
        t += '  ' + str(self.study_folder) + '\n'
        t += '  recordings: {} {}\n'.format(len(self.rec_names), self.rec_names)
        if len(self.sorter_names):
            t += '  sorters: {} {}\n'.format(len(self.sorter_names), self.sorter_names)

        return t

    def scan_folder(self):
        self.rec_names = get_rec_names(self.study_folder)
        # scan computed names
        self.computed_names = list(iter_computed_names(self.study_folder))  # list of pair (rec_name, sorter_name)
        self.sorter_names = np.unique([e for _, e in iter_computed_names(self.study_folder)]).tolist()
        self._is_scanned = True

    @classmethod
    def create(cls, study_folder, gt_dict):
        setup_comparison_study(study_folder, gt_dict)
        return cls(study_folder)

    def run_sorters(self, sorter_list, mode_if_folder_exists='keep', **kargs):
        #~ run_study_sorters(self.study_folder, sorter_list, sorter_params=sorter_params,
                          #~ mode=mode, engine=engine, engine_kwargs=engine_kwargs, verbose=verbose,
                          #~ run_sorter_kwargs=run_sorter_kwargs)

        sorter_folders = self.study_folder / 'sorter_folders'
        recording_dict = get_recordings(self.study_folder)
        
        print(recording_dict)
        
        run_sorters(sorter_list, recording_dict, sorter_folders, 
                with_output=False, mode_if_folder_exists=mode_if_folder_exists, **kargs)

        # results are copied so the heavy sorter_folders can be removed
        self.copy_sortings()





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
        #~ sorting = se.NpzSortingExtractor(self.study_folder / 'ground_truth' / (rec_name + '.npz'))
        sorting = load_extractor(self.study_folder /  'ground_truth' / rec_name)
        return sorting

    def get_recording(self, rec_name=None):
        rec_name = self._check_rec_name(rec_name)
        #~ rec = get_one_recording(self.study_folder, rec_name)
        rec = load_extractor(study_folder /  'raw_files' / rec_name)
        
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
        
        sorter_folders = self.study_folder / 'sorter_folders'
        sorting_folders = self.study_folder / 'sortings'
        log_olders = self.study_folder / 'sortings' / 'run_log'

        log_olders.mkdir(parents=True, exist_ok=True)

        for rec_name, sorter_name, output_folder in iter_output_folders(sorter_folders):
            SorterClass = sorter_dict[sorter_name]
            fname = rec_name + '[#]' + sorter_name
            npz_filename = sorting_folders / (fname + '.npz')

            sorting = SorterClass.get_result_from_folder(output_folder)
            try:
                sorting = SorterClass.get_result_from_folder(output_folder)
                NpzSortingExtractor.write_sorting(sorting, npz_filename)
            except:
                if npz_filename.is_file():
                    npz_filename.unlink()
            if (output_folder / 'spikeinterface_log.json').is_file():
                shutil.copyfile(output_folder / 'spikeinterface_log.json', sorting_folders / 'run_log' / (fname + '.json'))
        
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

    def aggregate_performance_by_units(self):
        assert self.comparisons is not None, 'run_comparisons first'

        perf_by_units = []
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            comp = self.comparisons[(rec_name, sorter_name)]
            
            perf = comp.get_performance(method='by_unit', output='pandas')
            perf['rec_name'] = rec_name
            perf['sorter_name'] = sorter_name
            perf = perf.reset_index()
            perf_by_units.append(perf)

        perf_by_units = pd.concat(perf_by_units)
        perf_by_units = perf_by_units.set_index(['rec_name', 'sorter_name', 'gt_unit_id'])

        return perf_by_units

    def aggregate_count_units(self, well_detected_score=None, redundant_score=None, overmerged_score=None):
        assert self.comparisons is not None, 'run_comparisons first'

        index = pd.MultiIndex.from_tuples(self.computed_names, names=['rec_name', 'sorter_name'])

        count_units = pd.DataFrame(index=index, columns=['num_gt', 'num_sorter', 'num_well_detected', 'num_redundant',
                                                         'num_overmerged'])

        if self.exhaustive_gt:
            count_units['num_false_positive'] = None
            count_units['num_bad'] = None

        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = self.comparisons[(rec_name, sorter_name)]

            count_units.loc[(rec_name, sorter_name), 'num_gt'] = len(gt_sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), 'num_sorter'] = len(sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), 'num_well_detected'] = \
                comp.count_well_detected_units(well_detected_score)
            if self.exhaustive_gt:
                count_units.loc[(rec_name, sorter_name), 'num_overmerged'] = \
                    comp.count_overmerged_units(overmerged_score)
                count_units.loc[(rec_name, sorter_name), 'num_redundant'] = \
                    comp.count_redundant_units(redundant_score)
                count_units.loc[(rec_name, sorter_name), 'num_false_positive'] = \
                    comp.count_false_positive_units(redundant_score)
                count_units.loc[(rec_name, sorter_name), 'num_bad'] = comp.count_bad_units()

        return count_units

    def aggregate_dataframes(self, copy_into_folder=True, **karg_thresh):
        dataframes = {}
        dataframes['run_times'] = self.aggregate_run_times().reset_index()
        perfs = self.aggregate_performance_by_units()

        dataframes['perf_by_units'] = perfs.reset_index()
        dataframes['count_units'] = self.aggregate_count_units(**karg_thresh).reset_index()

        if copy_into_folder:
            tables_folder = self.study_folder / 'tables'
            tables_folder.mkdir(parents=True, exist_ok=True)

            for name, df in dataframes.items():
                df.to_csv(str(tables_folder / (name + '.csv')), sep='\t', index=False)

        return dataframes

    def _compute_snr(self, rec_name, **snr_kargs):
        # TODO
        raise NotImplementedError        
        
        # print('compute SNR', rec_name)
        rec = self.get_recording(rec_name)
        gt_sorting = self.get_ground_truth(rec_name)

        snr_list = st.validation.compute_snrs(gt_sorting, rec, unit_ids=None, save_as_property=False, **snr_kargs)

        snr = pd.DataFrame(index=gt_sorting.get_unit_ids(), columns=['snr'])
        snr.index.name = 'gt_unit_id'
        snr.loc[:, 'snr'] = snr_list

        return snr

    def get_units_snr(self, rec_name=None, **snr_kargs):
        """
        Load or compute units SNR for a given recording.
        """
        rec_name = self._check_rec_name(rec_name)

        metrics_folder = self.study_folder / 'metrics'
        metrics_folder.mkdir(parents=True, exist_ok=True)

        filename = metrics_folder / ('SNR ' + rec_name + '.txt')

        if filename.is_file():
            snr = pd.read_csv(filename, sep='\t', index_col=None)
            snr = snr.set_index('gt_unit_id')
        else:
            snr = self._compute_snr(rec_name, **snr_kargs)
            snr.reset_index().to_csv(filename, sep='\t', index=False)
        snr['rec_name'] = rec_name
        return snr
    
    def concat_all_snr(self):
        snr = []
        for rec_name in self.rec_names:
            df = self.get_units_snr(rec_name)
            df = df.reset_index()
            snr.append(df)
        snr = pd.concat(snr)
        snr = snr.set_index(['rec_name', 'gt_unit_id'])
        return snr
