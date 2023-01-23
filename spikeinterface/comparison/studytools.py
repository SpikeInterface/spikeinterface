"""
High level tools to run many ground-truth comparison with
many sorter on many recordings and then collect and aggregate results
in an easy way.

The all mechanism is based on an intrinsic organization
into a "study_folder" with several subfolder:
  * raw_files : contain a copy in binary format of recordings
  * sorter_folders : contains output of sorters
  * ground_truth : contains a copy of sorting ground  in npz format
  * sortings: contains light copy of all sorting in npz format
  * tables: some table in cvs format
"""

from pathlib import Path
import shutil
import json
import os

import pandas as pd

from spikeinterface.core import load_extractor
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.extractors import NpzSortingExtractor
from spikeinterface.sorters import sorter_dict
from spikeinterface.sorters.launcher import iter_working_folder, iter_sorting_output

from .comparisontools import _perf_keys
from .paircomparisons import compare_sorter_to_ground_truth


def setup_comparison_study(study_folder, gt_dict, **job_kwargs):
    """
    Based on a dict of (recording, sorting) create the study folder.

    Parameters
    ----------
    study_folder: str
        The study folder.
    gt_dict : a dict of tuple (recording, sorting_gt)
        Dict of tuple that contain recording and sorting ground truth
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    study_folder = Path(study_folder)
    assert not study_folder.is_dir(), "'study_folder' already exists. Please remove it"

    study_folder.mkdir(parents=True, exist_ok=True)
    sorting_folders = study_folder / 'sortings'
    log_folder = sorting_folders / 'run_log'
    log_folder.mkdir(parents=True, exist_ok=True)
    tables_folder = study_folder / 'tables'
    tables_folder.mkdir(parents=True, exist_ok=True)

    for rec_name, (recording, sorting_gt) in gt_dict.items():
        # write recording using save with binary
        folder = study_folder / 'ground_truth' / rec_name
        sorting_gt.save(folder=folder, format='npz')
        folder = study_folder / 'raw_files' / rec_name
        recording.save(folder=folder, format='binary', **job_kwargs)

    # make an index of recording names
    with open(study_folder / 'names.txt', mode='w', encoding='utf8') as f:
        for rec_name in gt_dict:
            f.write(rec_name + '\n')


def get_rec_names(study_folder):
    """
    Get list of keys of recordings.
    Read from the 'names.txt' file in study folder.

    Parameters
    ----------
    study_folder: str
        The study folder.

    Returns
    -------
    rec_names: list
        List of names.
    """
    study_folder = Path(study_folder)
    with open(study_folder / 'names.txt', mode='r', encoding='utf8') as f:
        rec_names = f.read()[:-1].split('\n')
    return rec_names


def get_recordings(study_folder):
    """
    Get ground recording as a dict.

    They are read from the 'raw_files' folder with binary format.

    Parameters
    ----------
    study_folder: str
        The study folder.

    Returns
    -------
    recording_dict: dict
        Dict of recording.
    """
    study_folder = Path(study_folder)

    rec_names = get_rec_names(study_folder)
    recording_dict = {}
    for rec_name in rec_names:
        rec = load_extractor(study_folder / 'raw_files' / rec_name)
        recording_dict[rec_name] = rec

    return recording_dict


def get_ground_truths(study_folder):
    """
    Get ground truth sorting extractor as a dict.

    They are read from the 'ground_truth' folder with npz format.

    Parameters
    ----------
    study_folder: str
        The study folder.

    Returns
    -------
    ground_truths: dict
        Dict of sorting_gt.
    """
    study_folder = Path(study_folder)
    rec_names = get_rec_names(study_folder)
    ground_truths = {}
    for rec_name in rec_names:
        sorting = load_extractor(study_folder / 'ground_truth' / rec_name)
        ground_truths[rec_name] = sorting
    return ground_truths


def iter_computed_names(study_folder):
    sorting_folder = Path(study_folder) / 'sortings'
    for filename in os.listdir(sorting_folder):
        if filename.endswith('.npz') and '[#]' in filename:
            rec_name, sorter_name = filename.replace('.npz', '').split('[#]')
            yield rec_name, sorter_name


def iter_computed_sorting(study_folder):
    """
    Iter over sorting files.
    """
    sorting_folder = Path(study_folder) / 'sortings'
    for filename in os.listdir(sorting_folder):
        if filename.endswith('.npz') and '[#]' in filename:
            rec_name, sorter_name = filename.replace('.npz', '').split('[#]')
            sorting = NpzSortingExtractor(sorting_folder / filename)
            yield rec_name, sorter_name, sorting


def collect_run_times(study_folder):
    """
    Collect run times in a working folder and store it in CVS files.

    The output is list of (rec_name, sorter_name, run_time)
    """
    study_folder = Path(study_folder)
    sorting_folders = study_folder / 'sortings'
    log_folder = sorting_folders / 'run_log'
    tables_folder = study_folder / 'tables'

    tables_folder.mkdir(parents=True, exist_ok=True)

    run_times = []
    for filename in os.listdir(log_folder):
        if filename.endswith('.json') and '[#]' in filename:
            rec_name, sorter_name = filename.replace('.json', '').split('[#]')
            with open(log_folder / filename, encoding='utf8', mode='r') as logfile:
                log = json.load(logfile)
                run_time = log.get('run_time', None)
            run_times.append((rec_name, sorter_name, run_time))

    run_times = pd.DataFrame(run_times, columns=['rec_name', 'sorter_name', 'run_time'])
    run_times = run_times.set_index(['rec_name', 'sorter_name'])

    return run_times


def aggregate_sorting_comparison(study_folder, exhaustive_gt=False):
    """
    Loop over output folder in a tree to collect sorting output and run
    ground_truth_comparison on them.

    Parameters
    ----------
    study_folder: str
        The study folder.
    exhaustive_gt: bool (default True)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurement.
        For instance, MEArec simulated dataset have exhaustive_gt=True

    Returns
    ----------
    comparisons: a dict of SortingComparison

    """

    study_folder = Path(study_folder)

    ground_truths = get_ground_truths(study_folder)
    results = collect_study_sorting(study_folder)

    comparisons = {}
    for (rec_name, sorter_name), sorting in results.items():
        gt_sorting = ground_truths[rec_name]
        sc = compare_sorter_to_ground_truth(gt_sorting, sorting, exhaustive_gt=exhaustive_gt)
        comparisons[(rec_name, sorter_name)] = sc

    return comparisons


def aggregate_performances_table(study_folder, exhaustive_gt=False, **karg_thresh):
    """
    Aggregate some results into dataframe to have a "study" overview on all recordingXsorter.

    Tables are:
      * run_times: run times per recordingXsorter
      * perf_pooled_with_sum: GroundTruthComparison.see get_performance
      * perf_pooled_with_average: GroundTruthComparison.see get_performance
      * count_units: given some threshold count how many units : 'well_detected', 'redundant', 'false_postive_units, 'bad'

    Parameters
    ----------
    study_folder: str
        The study folder.
    karg_thresh: dict
        Threshold parameters used for the "count_units" table.

    Returns
    -------
    dataframes: a dict of DataFrame
        Return several useful DataFrame to compare all results.
        Note that count_units depend on karg_thresh.
    """
    study_folder = Path(study_folder)
    sorter_folders = study_folder / 'sorter_folders'
    tables_folder = study_folder / 'tables'

    comparisons = aggregate_sorting_comparison(study_folder, exhaustive_gt=exhaustive_gt)
    ground_truths = get_ground_truths(study_folder)
    results = collect_study_sorting(study_folder)

    study_folder = Path(study_folder)

    dataframes = {}

    # get run times:
    run_times = pd.read_csv(str(tables_folder / 'run_times.csv'), sep='\t')
    run_times.columns = ['rec_name', 'sorter_name', 'run_time']
    run_times = run_times.set_index(['rec_name', 'sorter_name', ])
    dataframes['run_times'] = run_times

    perf_pooled_with_sum = pd.DataFrame(index=run_times.index, columns=_perf_keys)
    dataframes['perf_pooled_with_sum'] = perf_pooled_with_sum

    perf_pooled_with_average = pd.DataFrame(index=run_times.index, columns=_perf_keys)
    dataframes['perf_pooled_with_average'] = perf_pooled_with_average

    count_units = pd.DataFrame(index=run_times.index,
                               columns=['num_gt', 'num_sorter', 'num_well_detected', 'num_redundant'])
    dataframes['count_units'] = count_units
    if exhaustive_gt:
        count_units['num_false_positive'] = None
        count_units['num_bad'] = None

    perf_by_spiketrain = []

    for (rec_name, sorter_name), comp in comparisons.items():
        gt_sorting = ground_truths[rec_name]
        sorting = results[(rec_name, sorter_name)]

        perf = comp.get_performance(method='pooled_with_sum', output='pandas')
        perf_pooled_with_sum.loc[(rec_name, sorter_name), :] = perf

        perf = comp.get_performance(method='pooled_with_average', output='pandas')
        perf_pooled_with_average.loc[(rec_name, sorter_name), :] = perf

        perf = comp.get_performance(method='by_spiketrain', output='pandas')
        perf['rec_name'] = rec_name
        perf['sorter_name'] = sorter_name
        perf = perf.reset_index()

        perf_by_spiketrain.append(perf)

        count_units.loc[(rec_name, sorter_name), 'num_gt'] = len(gt_sorting.get_unit_ids())
        count_units.loc[(rec_name, sorter_name), 'num_sorter'] = len(sorting.get_unit_ids())
        count_units.loc[(rec_name, sorter_name), 'num_well_detected'] = comp.count_well_detected_units(**karg_thresh)
        count_units.loc[(rec_name, sorter_name), 'num_redundant'] = comp.count_redundant_units()
        if exhaustive_gt:
            count_units.loc[(rec_name, sorter_name), 'num_false_positive'] = comp.count_false_positive_units()
            count_units.loc[(rec_name, sorter_name), 'num_bad'] = comp.count_bad_units()

    perf_by_spiketrain = pd.concat(perf_by_spiketrain)
    perf_by_spiketrain = perf_by_spiketrain.set_index(['rec_name', 'sorter_name', 'gt_unit_id'])
    dataframes['perf_by_spiketrain'] = perf_by_spiketrain

    return dataframes
