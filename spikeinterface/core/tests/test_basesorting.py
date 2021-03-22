"""
test for BaseSorting are done with NpzSortingExtractor.
but check only for BaseRecording general methods.
"""
import shutil
from pathlib import Path
import pytest
import numpy as np

from spikeinterface.core import NpzSortingExtractor, load_extractor
from spikeinterface.core.base import BaseExtractor

from spikeinterface.core.tests.testing_tools import create_sorting_npz

def _clean_all():
    cache_folder = './my_cache_folder'
    if Path(cache_folder).exists():
        shutil.rmtree(cache_folder)
    
def setup_module():
    _clean_all()

def teardown_module():
    _clean_all()



def test_BaseSorting():
    num_seg = 2
    file_path = 'test_BaseSorting.npz'
    
    create_sorting_npz(num_seg, file_path)
    
    sorting = NpzSortingExtractor(file_path)
    print(sorting)


    assert sorting.get_num_segments() == 2
    assert sorting.get_num_units() == 3
    
    
    # annotations / properties
    sorting.annotate(yep='yop')
    assert sorting.get_annotation('yep') == 'yop'
    
    sorting.set_property('amplitude', [-20, -40., -55.5])
    values = sorting.get_property('amplitude')
    assert np.all(values ==  [-20, -40., -55.5])
    
    # dump/load dict
    d = sorting.to_dict()
    sorting2 = BaseExtractor.from_dict(d)
    sorting3 = load_extractor(d)
    
    # dump/load json
    sorting.dump_to_json('test_BaseSorting.json')
    sorting2 = BaseExtractor.load('test_BaseSorting.json')
    sorting3 = load_extractor('test_BaseSorting.json')

    # dump/load pickle
    sorting.dump_to_pickle('test_BaseSorting.pkl')
    sorting2 = BaseExtractor.load('test_BaseSorting.pkl')
    sorting3 = load_extractor('test_BaseSorting.pkl')

    # cache
    cache_folder = './my_cache_folder'
    sorting.set_cache_folder(cache_folder)
    sorting.cache(name='simple_sorting')
    sorting2 = BaseExtractor.load_from_cache(cache_folder, 'simple_sorting')
    # but also possible
    sorting3 = BaseExtractor.load('./my_cache_folder/simple_sorting')


if __name__ == '__main__':
    _clean_all()
    test_BaseSorting()

