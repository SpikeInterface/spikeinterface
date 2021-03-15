import os
import shutil
import time
import pickle

import pytest

from spikeinterface.extractors import toy_example
from spikeinterface.comparison.studytools import (setup_comparison_study,
    iter_computed_names, iter_computed_sorting, 
    get_rec_names, get_ground_truths, get_recordings)


study_folder = 'test_studytools/'


def setup_module():
    if os.path.exists(study_folder):
        shutil.rmtree(study_folder)


def test_setup_comparison_study():
    rec0, gt_sorting0 = toy_example(num_channels=4, duration=30, seed=0, num_segments=1)
    rec1, gt_sorting1 = toy_example(num_channels=32, duration=30, seed=0, num_segments=1)
    
    gt_dict = {
        'toy_tetrode': (rec0, gt_sorting0),
        'toy_probe32': (rec1, gt_sorting1),
    }
    setup_comparison_study(study_folder, gt_dict)


def test_get_ground_truths():
    names = get_rec_names(study_folder)
    d = get_ground_truths(study_folder)
    d = get_recordings(study_folder)



def test_loops():
    names = list(iter_computed_names(study_folder))
    for rec_name, sorter_name, sorting in iter_computed_sorting(study_folder):
        print(rec_name, sorter_name)
        print(sorting)


if __name__ == '__main__':
    setup_module()
    test_setup_comparison_study()
    test_get_ground_truths()
    test_loops()
