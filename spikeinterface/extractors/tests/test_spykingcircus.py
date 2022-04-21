import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


@pytest.mark.skip('SpykingCIRCUS can be tested after running run_spykingcircus()')
def test_spykingcircusextractors():
    # not tested here, tested in run_spykingcircus(...)
    pass

    #  sc_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/spykingcircus_output/'
    # sorting = SpykingCircusSortingExtractor(sc_folder)
    #  print(sorting)


if __name__ == '__main__':
    test_spykingcircusextractors()
