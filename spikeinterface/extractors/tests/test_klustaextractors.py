import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import *


def test_klustaextractors():
    klusta_folder = '/home/samuel/Documents/SpikeInterface/spikeinterface/spikeinterface/sorters/tests/klusta_output/'
    sorting = KlustaSortingExtractor(klusta_folder)
    print(sorting)

    
if __name__ == '__main__':
    test_klustaextractors()
