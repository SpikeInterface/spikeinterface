import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


@pytest.mark.skip('WaveClus can be tested after running run_waveclus()')
def test_waveclustextractors():
    # not tested here, tested in run_waveclus(...)
    pass

    #  wc_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/waveclus_output/times_results.mat'
    #  sorting = WaveClusSortingExtractor(wc_folder)
    #  print(sorting)


if __name__ == '__main__':
    test_waveclustextractors()
