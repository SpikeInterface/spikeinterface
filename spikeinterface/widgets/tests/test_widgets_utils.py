import unittest
import sys

if __name__ != '__main__':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spikeinterface import extract_waveforms, download_dataset
import spikeinterface.extractors as se

from spikeinterface.widgets.utils import get_unit_colors


def test_get_unit_colors():
        local_path = download_dataset(remote_path='mearec/mearec_test_10s.h5')
        rec = se.MEArecRecordingExtractor(local_path)
        sorting = se.MEArecSortingExtractor(local_path)
    
        colors = get_unit_colors(sorting)
        print(colors)
        

if __name__ == '__main__':
    test_get_unit_colors()
    
    
