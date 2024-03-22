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
    traces_in0 = rec0.get_traces(segment_index=0, start_frame=0, end_frame=1000)
    traces_in1 = rec0.get_traces(segment_index=0, start_frame=5000, end_frame=6000)
    traces_out0 = rec0.get_traces(segment_index=0, start_frame=2000, end_frame=3000)
    assert np.all(traces_in0 == 0)
    assert np.all(traces_in1 == 0)
    assert not np.all(traces_out0 == 0)

    rec1 = silence_periods(rec, list_periods=[[[0, 1000], [5000, 6000]], []], mode="noise")
    rec1 = rec1.save(folder=cache_folder / "rec_w_noise", verbose=False, overwrite=True)
    noise_levels = get_noise_levels(rec, return_scaled=False)
    traces_in0 = rec1.get_traces(segment_index=0, start_frame=0, end_frame=1000)
    traces_in1 = rec1.get_traces(segment_index=0, start_frame=5000, end_frame=6000)
    assert np.abs((np.std(traces_in0, axis=0) - noise_levels) < 0.1).sum()
    assert np.abs((np.std(traces_in1, axis=0) - noise_levels)).sum() < 0.1
    data1 = rec.get_traces(0, 400, 600)
    data2 = rec.get_traces(0, 500, 700)
    assert np.all(data1[100:] == data2[:100])

    traces_mix = rec0.get_traces(segment_index=0, start_frame=900, end_frame=5100)
    traces_original = rec.get_traces(segment_index=0, start_frame=900, end_frame=5100)
    assert np.all(traces_original[100:-100] == traces_mix[100:-100])
    assert np.all(traces_mix[:100] == 0)
    assert np.all(traces_mix[-100:] == 0)
    assert not np.all(traces_mix[:200] == 0)
    assert not np.all(traces_mix[:-200] == 0)


if __name__ == "__main__":
    test_silence()
