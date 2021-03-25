import unittest
import pytest

from spikeinterface.core.tests.testing_tools import generate_recording

from spikeinterface.toolkit.preprocessing import clip,  blank_staturation

import numpy as np

def test_clip():
    rec = generate_recording()

    
    rec2 = clip(rec, a_min=-2, a_max=3.)
    rec2.save(verbose=False)
    
    rec3 = clip(rec, a_min=-1.5)
    rec3.save(verbose=False)
    
    traces = rec2.get_traces(segment_index=0, channel_ids=[1])
    assert traces.shape[1] == 1

    #~ import matplotlib.pyplot as plt
    #~ from spikeinterface.widgets import plot_timeseries
    #~ fig, ax = plt.subplots()
    #~ ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    #~ ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    #~ ax.plot(rec3.get_traces(segment_index=0)[:, 0], color='y')
    #~ plt.show()

    


def test_blank_staturationy():
    rec = generate_recording()
    
    rec2 = blank_staturation(rec, abs_threshold=3.)
    rec2.save(verbose=False)
    
    rec3 = blank_staturation(rec, quantile_threshold=0.01, direction='both')
    rec3.save(verbose=False)
    
    traces = rec2.get_traces(segment_index=0, channel_ids=[1])
    assert traces.shape[1] == 1

    #~ import matplotlib.pyplot as plt
    #~ from spikeinterface.widgets import plot_timeseries
    #~ fig, ax = plt.subplots()
    #~ ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    #~ ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    #~ ax.plot(rec3.get_traces(segment_index=0)[:, 0], color='y')
    #~ plt.show()


if __name__ == '__main__':
    test_clip()
    test_blank_staturationy()
