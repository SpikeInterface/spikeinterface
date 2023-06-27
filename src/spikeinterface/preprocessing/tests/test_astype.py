import pytest
from pathlib import Path
import shutil

from spikeinterface import set_global_tmp_folder, NumpyRecording
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import astype

import numpy as np


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)


def test_astype():
    rng = np.random.RandomState(0)
    traces = (rng.randn(10000, 4) * 100).astype("float32")
    rec_float32 = NumpyRecording(traces, sampling_frequency=30000)
    traces_int16 = traces.astype("int16")
    np.testing.assert_array_equal(traces_int16, astype(rec_float32, "int16").get_traces())
    traces_float64 = traces.astype("float64")
    np.testing.assert_array_equal(traces_float64, astype(rec_float32, "float64").get_traces())


if __name__ == "__main__":
    test_astype()
