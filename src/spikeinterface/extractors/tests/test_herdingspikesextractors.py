import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


@pytest.mark.skip('HerdingSpikes can be tested after running run_herdingspikes()')
def test_herdingspikesextractors():
    # no tested here, tested un run_herdingspikes()
    pass

    #  hs_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/herdingspikes_output/HS2_sorted.hdf5'
    #  sorting = HerdingspikesSortingExtractor(hs_folder)
    #  print(sorting)


if __name__ == '__main__':
    test_herdingspikesextractors()
