import unittest
import pytest

from spikeinterface.toolkit.preprocessing.tests.testing_tools import generate_recording

from spikeinterface.toolkit.preprocessing import normalize_by_quantile, scale, center

import numpy as np


def test_normalize_by_quantile():
    rec = generate_recording()
    
    rec2 = normalize_by_quantile(rec, mode='by_channel')
    rec2.save(verbose=False)
    
    traces = rec2.get_traces(segment_index=0, channel_ids=[1])
    assert traces.shape[1] == 1
    
    rec2 = normalize_by_quantile(rec, mode='pool_channel')
    rec2.save(verbose=False)
    
    
    
    
    #~ import matplotlib.pyplot as plt
    #~ from spikeinterface.widgets import plot_timeseries
    #~ fig, ax = plt.subplots()
    #~ ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    #~ ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    #~ plt.show()

def test_scale():
    rec = generate_recording()
    n = rec.get_num_channels()
    gain = np.ones(n) * 2.
    offset = np.ones(n) * -10.
    
    rec2 = scale(rec, gain=gain, offset=offset)
    rec2.get_traces(segment_index=0)

    rec2 = scale(rec, gain=2., offset=-10.)
    rec2.get_traces(segment_index=0)

    rec2 = scale(rec, gain=gain, offset=-10.)
    rec2.get_traces(segment_index=0)


def test_center():
    rec = generate_recording()
    
    rec2 = center(rec, mode='median')
    rec2.get_traces(segment_index=0)



if __name__ == '__main__':
    test_normalize_by_quantile()
    
    test_scale()
    test_center()
