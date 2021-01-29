"""
test for BaseSorting are done with NpzSortingExtractor.
but check only for BaseRecording general methods.
"""
import pytest
import numpy as np

from spikeinterface.core import NpzSortingExtractor


def test_BaseSorting():
    sorting = NpzSortingExtractor()
    print(sorting)


if __name__ == '__main__':
    test_BaseSorting()

