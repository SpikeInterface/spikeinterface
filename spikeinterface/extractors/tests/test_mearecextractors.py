import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import MEArecRecordingExtractor,MEArecSortingExtractor

from spikeinterface.extractors.tests.file_retrieve import download_test_file

def test_mearec_extractors():
    mearec_file = download_test_file('mearec', 'mearec_test_10s.h5', local_folder='extractor_testing_files')
    
    rec = MEArecRecordingExtractor(mearec_file)
    print(rec)
    
    sorting = MEArecSortingExtractor(mearec_file, use_natural_unit_ids=True)
    print(sorting)
    print(sorting.get_unit_ids())

    sorting = MEArecSortingExtractor(mearec_file, use_natural_unit_ids=False)
    print(sorting)
    print(sorting.get_unit_ids())
    
    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    #~ traces = rec.get_traces()
    #~ ax.plot(traces[:32000*4, 1])
    #~ plt.show()
    

if __name__ == '__main__':
    test_mearec_extractors()
