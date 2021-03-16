import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_herdingspikesextractors():
    hs_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/herdingspikes_output/HS2_sorted.hdf5'
    sorting = HerdingspikesSortingExtractor(hs_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_herdingspikesextractors()
