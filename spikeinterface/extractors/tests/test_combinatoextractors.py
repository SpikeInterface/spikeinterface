import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_combinatoextractors():
    combinato_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/combinato_output/recording'
    sorting = CombinatoSortingExtractor(combinato_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_combinatoextractors()
