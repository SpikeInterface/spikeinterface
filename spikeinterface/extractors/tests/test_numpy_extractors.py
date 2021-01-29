import shutil
from pathlib import Path

import pytest
import numpy as np

from spikeinterface.extractors import NumpyRecording,NumpySorting


def _clean_all():
    cache_folder = './my_cache_folder'
    if Path(cache_folder).exists():
        shutil.rmtree(cache_folder)
    
def setup_module():
    _clean_all()

def teardown_module():
    _clean_all()

def test_NumpyRecording():
    sampling_frequency = 30000
    timeseries_list = []
    for seg_index in range(3):
        traces = np.zeros((1000, 5), dtype='float64')
        timeseries_list.append(traces)
    
    rec = NumpyRecording(timeseries_list, sampling_frequency)
    print(rec)

    cache_folder = './my_cache_folder'
    rec.set_cache_folder(cache_folder)
    rec.cache(name='test_NumpyRecording')
    

#~ # TODO
#~ def test_NumpySorting():
    #~ sorting = NumpySorting()
    #~ print(sorting)
    
    

if __name__ == '__main__':
    _clean_all()
    test_NumpyRecording()
    #~ test_NumpySorting()
