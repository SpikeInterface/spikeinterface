import pytest
import numpy as np

from spikeinterface.extractors import toy_example



def test_toy_example():
    rec, sorting = toy_example(num_segments=2, num_units=10)
    assert rec.get_num_segments() == 2
    assert sorting.get_num_segments() == 2
    assert sorting.get_num_units() == 10
    # print(rec)
    # print(sorting)
    
    rec, sorting = toy_example(num_segments=1)
    assert rec.get_num_segments() == 1
    assert sorting.get_num_segments() == 1
    print(rec)
    print(sorting)
    
    # print(rec.get_channel_locations())
    
    probe = rec.get_probe()
    print(probe)
    
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(rec.get_traces())
    # plt.show()

    
if __name__ == '__main__':
    test_toy_example()
