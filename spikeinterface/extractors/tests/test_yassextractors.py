import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_yassextractors():
    yass_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/yass_output/'
    sorting = YassSortingExtractor(yass_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_yassextractors()
