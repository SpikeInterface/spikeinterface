import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


@pytest.mark.skip('')
def test_yassextractors():
    # not tested here, tested in run_yass(...)
    pass
    
    # yass_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/yass_output/'
    # sorting = YassSortingExtractor(yass_folder)
    # print(sorting)

    
if __name__ == '__main__':
    test_yassextractors()
