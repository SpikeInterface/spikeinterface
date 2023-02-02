import pytest
from pathlib import Path
import shutil

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import silence_periods

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

    traces0 = rec0.get_traces(segment_index=0, channel_ids=[1])

    rec1 = silence_periods(rec, list_periods=[[[0, 1000], [5000, 6000]], []], mode="noise")
    rec1.save(verbose=False)



if __name__ == '__main__':
    test_silence()
