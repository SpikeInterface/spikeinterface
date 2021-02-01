import pytest
import numpy as np

from spikeinterface.extractors import toy_example



def test_toy_example():
    rec, sorting = toy_example(num_segments=2)
    assert rec.get_num_segments() == 2
    assert sorting.get_num_segments() == 2
    #~ print(rec)
    #~ print(sorting)
    
    rec, sorting = toy_example(num_segments=1)
    assert rec.get_num_segments() == 1
    assert sorting.get_num_segments() == 1
    #~ print(rec)
    #~ print(sorting)
    
    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    #~ ax.plot(rec.get_traces().T)
    #~ plt.show()

    
if __name__ == '__main__':
    test_toy_example()
