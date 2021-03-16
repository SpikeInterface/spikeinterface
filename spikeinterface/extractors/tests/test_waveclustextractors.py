import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_waveclustextractors():
    hdsort_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/waveclus_output/times_results.mat'
    sorting = WaveClusSortingExtractor(hdsort_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_waveclustextractors()
