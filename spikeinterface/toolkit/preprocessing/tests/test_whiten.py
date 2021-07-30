import unittest
import pytest

from spikeinterface.core.tests.testing_tools import generate_recording

from spikeinterface.toolkit.preprocessing import whiten


def test_normalize_by_quantile():
    rec = generate_recording()

    rec2 = whiten(rec)
    rec2.save(verbose=False)

    # ~ import matplotlib.pyplot as plt
    # ~ from spikeinterface.widgets import plot_timeseries
    # ~ fig, ax = plt.subplots()
    # ~ ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    # ~ ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    # ~ plt.show()


if __name__ == '__main__':
    test_normalize_by_quantile()
