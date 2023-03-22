import pytest
from pathlib import Path

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import normalize_by_quantile, scale, center, zscore

import numpy as np


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)


def test_normalize_by_quantile():
    rec = generate_recording()

    rec2 = normalize_by_quantile(rec, mode='by_channel')
    rec2.save(verbose=False)

    traces = rec2.get_traces(segment_index=0, channel_ids=[1])
    assert traces.shape[1] == 1

    rec2 = normalize_by_quantile(rec, mode='pool_channel')
    rec2.save(verbose=False)

    # import matplotlib.pyplot as plt
    # from spikeinterface.widgets import plot_timeseries
    # fig, ax = plt.subplots()
    # ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    # ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    # plt.show()


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
    

def test_zscore():
    rec = generate_recording()
    tr = rec.get_traces(segment_index=0)

    rec2 = zscore(rec)
    tr = rec2.get_traces(segment_index=0)
    meds = np.median(tr, axis=0)
    mads = np.median(np.abs(tr - meds), axis=0) / 0.6744897501960817
    assert np.all(np.abs(meds) < 0.01)
    assert np.all(np.abs(mads - 1) < 0.01)
    assert 'gain' in rec2._kwargs
    
    rec3 = zscore(rec, mode="mean+std")
    tr = rec3.get_traces(segment_index=0)
    assert np.all(np.abs(np.mean(tr, axis=0)) < 0.01)
    assert np.all(np.abs(np.std(tr, axis=0) - 1) < 0.01)

    rec_int = scale(rec, dtype="int16", gain=100)
    with pytest.raises(AssertionError):
        rec4 = zscore(rec_int, dtype=None)
    rec4 = zscore(rec_int, dtype='float32', mode="mean+std")
    rec4 = zscore(rec_int, dtype='int16', int_scale=256, mode="mean+std")
    tr = rec4.get_traces(segment_index=0)
    assert np.all(np.abs(np.mean(tr, axis=0)) < 1)
    assert np.all(np.abs(np.std(tr, axis=0) - 256) < 1)



if __name__ == '__main__':
    test_zscore()
