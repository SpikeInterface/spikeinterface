import pytest
import numpy as np

from spikeinterface.extractors import NumpyRecording,NumpySorting


def test_NumpyRecording():
    rec = NumpyRecording()
    print(rec)

def test_NumpySorting():
    sorting = NumpySorting()
    print(sorting)
    
    

if __name__ == '__main__':
    test_NumpyRecording()
    test_NumpySorting()
