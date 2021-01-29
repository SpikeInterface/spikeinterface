import pytest
import numpy as np

from spikeinterface.extractors import toy_example



def test_toy_example():
    rec, sorting = toy_example(num_segments=2)
    assert rec.get_num_segments() == 2
    assert sorting.get_num_segments() == 2
    
    rec, sorting = toy_example(num_segments=1)
    assert rec.get_num_segments() == 1
    assert sorting.get_num_segments() == 1
    
    
    
    
    
if __name__ == '__main__':
    test_toy_example()
