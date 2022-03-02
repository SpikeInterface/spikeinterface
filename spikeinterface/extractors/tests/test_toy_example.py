import pytest
import numpy as np

from spikeinterface.extractors import toy_example


def test_toy_example():
    rec, sorting = toy_example(num_segments=2, num_units=10)
    assert rec.get_num_segments() == 2
    assert sorting.get_num_segments() == 2
    assert sorting.get_num_units() == 10

    rec, sorting = toy_example(num_segments=1, num_channels=16, num_columns=2)
    assert rec.get_num_segments() == 1
    assert sorting.get_num_segments() == 1
    print(rec)
    print(sorting)

    probe = rec.get_probe()
    print(probe)


if __name__ == '__main__':
    test_toy_example()
