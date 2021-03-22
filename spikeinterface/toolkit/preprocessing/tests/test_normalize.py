import unittest
import pytest

from spikeinterface.toolkit.preprocessing.tests.testing_tools import generate_recording

from spikeinterface.toolkit.preprocessing import normalize_by_quantile



def test_normalize_by_quantile():
    rec = generate_recording()
    
    rec2 = normalize_by_quantile(rec,  )
    
    
    
    # import matplotlib.pyplot as plt
    # from spikeinterface.widgets import plot_timeseries
    # plot_timeseries(rec, segment_index=0)
    # plot_timeseries(rec2, segment_index=0)
    # plot_timeseries(rec3, segment_index=0)
    # plot_timeseries(rec4, segment_index=0)
    # plt.show()


if __name__ == '__main__':
    test_normalize_by_quantile()
