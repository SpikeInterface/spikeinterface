import pytest
from pathlib import Path

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import rectify

import numpy as np


def test_rectify():
    rec = generate_recording()

    rec2 = rectify(rec)
    rec2.save(verbose=False)

    traces = rec2.get_traces(segment_index=0, channel_ids=[1])
    assert traces.shape[1] == 1

    # import matplotlib.pyplot as plt
    # from spikeinterface.widgets import plot_traces
    # fig, ax = plt.subplots()
    # ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    # ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    # plt.show()


if __name__ == "__main__":
    test_rectify()
