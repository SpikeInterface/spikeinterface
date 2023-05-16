import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


@pytest.mark.skip('Tridesclous can be tested after running run_tridesclous()')
def test_tridesclousextractors():
    # not tested here, tested in run_tridesclous(...)
    pass

    #  tdc_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/tridesclous_output/'
    #  sorting = TridesclousSortingExtractor(tdc_folder)
    #  print(sorting)


if __name__ == '__main__':
    test_tridesclousextractors()
