import os
import shutil
import time
import pickle

import pytest


from spikeinterface.extractors import toy_example
from spikeinterface.sorters import installed_sorters
from spikeinterface.comparison import GroundTruthStudy

study_folder = 'test_groundtruthstudy/'


def setup_module():
    if os.path.exists(study_folder):
        shutil.rmtree(study_folder)
    _setup_comparison_study()


def _setup_comparison_study():
    rec0, gt_sorting0 = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
    rec1, gt_sorting1 = toy_example(num_channels=32, duration=30, seed=0, num_segments=1)
    
    gt_dict = {
        'toy_tetrode': (rec0, gt_sorting0),
        'toy_probe32': (rec1, gt_sorting1),
    }
    study = GroundTruthStudy.create(study_folder, gt_dict)


def test_run_study_sorters():
    study = GroundTruthStudy(study_folder)
    sorter_list = ['tridesclous', 'spykingcircus']
    print(f"\n#################################\nINSTALLED SORTERS\n#################################\n"
          f"{installed_sorters()}")
    study.run_sorters(sorter_list)


def test_extract_sortings():
    study = GroundTruthStudy(study_folder)
    print(study)
    
    study.copy_sortings()
    
    exit()

    for rec_name in study.rec_names:
        gt_sorting = study.get_ground_truth(rec_name)
        # ~ print(rec_name, gt_sorting)

    for rec_name in study.rec_names:
        snr = study.get_units_snr(rec_name=rec_name)
        # print(snr)

    study.copy_sortings()
    study.run_comparisons(exhaustive_gt=True)

    run_times = study.aggregate_run_times()
    perf = study.aggregate_performance_by_units()
    count_units = study.aggregate_count_units()
    dataframes = study.aggregate_dataframes()

    shutil.rmtree(study_folder)


if __name__ == '__main__':
    setup_module()
    test_run_study_sorters()
    test_extract_sortings()
