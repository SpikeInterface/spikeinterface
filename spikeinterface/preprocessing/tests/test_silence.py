import pytest
from pathlib import Path
import shutil

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import silence_periods


from spikeinterface.core import get_noise_levels

import numpy as np


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)

def test_silence():
    rec = generate_recording()

    rec0 = silence_periods(rec, list_periods=[[[0, 1000], [5000, 6000]], []], mode="zeros")
    rec0.save(verbose=False)

    traces0 = rec0.get_traces(segment_index=0, start_frame=0, end_frame=1000)
    traces1 = rec0.get_traces(segment_index=0, start_frame=5000, end_frame=6000)
    assert np.all(traces0 == 0) and np.all(traces1 == 0)

    rec1 = silence_periods(rec, list_periods=[[[0, 1000], [5000, 6000]], []], mode="noise")
    rec1.save(verbose=False)
    noise_levels = get_noise_levels(rec, return_scaled=False)
    traces0 = rec1.get_traces(segment_index=0, start_frame=0, end_frame=1000)
    traces1 = rec1.get_traces(segment_index=0, start_frame=5000, end_frame=6000)
    assert np.abs((np.std(traces0, axis=0) - noise_levels) < 0.1).sum() and np.abs((np.std(traces1, axis=0) - noise_levels)).sum() < 0.1



if __name__ == '__main__':
    test_silence()
