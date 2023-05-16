import pytest
from pathlib import Path
import shutil

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import clip, blank_staturation

import numpy as np


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)


def test_clip():
    rec = generate_recording()

    rec0 = clip(rec, a_min=-2, a_max=3.)
    rec0.save(verbose=False)

    rec1 = clip(rec, a_min=-1.5)
    rec1.save(verbose=False)

    traces0 = rec0.get_traces(segment_index=0, channel_ids=[1])
    assert traces0.shape[1] == 1

    assert np.all(-2 <= traces0[0] <= 3)

    traces1 = rec1.get_traces(segment_index=0, channel_ids=[0, 1])
    assert traces1.shape[1] == 2

    assert np.all(-1.5 <= traces1[1])


def test_blank_staturation():
    rec = generate_recording()

    rec0 = blank_staturation(rec, abs_threshold=3.)
    rec0.save(verbose=False)

    rec1 = blank_staturation(rec, quantile_threshold=0.01, direction='both',
                             chunk_size=10000)
    rec1.save(verbose=False)

    traces0 = rec0.get_traces(segment_index=0, channel_ids=[1])
    assert traces0.shape[1] == 1
    assert np.all(traces0 < 3.)

    traces1 = rec1.get_traces(segment_index=0, channel_ids=[0])
    assert traces1.shape[1] == 1
    # use a smaller value to be sure
    a_min = rec1._recording_segments[0].a_min
    assert np.all(traces1 >= a_min)


if __name__ == '__main__':
    test_clip()
    test_blank_staturation()
