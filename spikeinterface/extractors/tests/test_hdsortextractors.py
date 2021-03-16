import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_klustaextractors():
    hdsort_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/hdsort_output/'
    sorting = HDSortSortingExtractor(hdsort_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_klustaextractors()
