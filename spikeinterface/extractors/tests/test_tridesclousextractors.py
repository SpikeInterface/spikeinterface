import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_tridesclousextractors():
    tdc_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/tridesclous_output/'
    sorting = TridesclousSortingExtractor(tdc_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_tridesclousextractors()
