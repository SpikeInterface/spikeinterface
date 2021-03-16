import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_spykingcircusextractors():
    tdc_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/spykingcircus_output/'
    sorting = SpykingCircusSortingExtractor(tdc_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_spykingcircusextractors()
