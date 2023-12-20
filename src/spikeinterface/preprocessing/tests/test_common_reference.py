import pytest
from pathlib import Path
import shutil

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import CommonReferenceRecording, common_reference

import numpy as np

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)


def test_common_reference():
    rec = generate_recording(durations=[5.0], num_channels=4)
    rec._main_ids = np.array(["a", "b", "c", "d"])

    # no groups
    rec_cmr = common_reference(rec, reference="global", operator="median")
    rec_car = common_reference(rec, reference="global", operator="average")
    rec_sin = common_reference(rec, reference="single", ref_channel_ids=["a"])
    rec_local_car = common_reference(rec, reference="local", local_radius=(20, 65), operator="median")

    rec_cmr.save(verbose=False)
    rec_car.save(verbose=False)
    rec_sin.save(verbose=False)
    rec_local_car.save(verbose=False)

    traces = rec.get_traces()
    assert np.allclose(traces, rec_cmr.get_traces() + np.median(traces, axis=1, keepdims=True), atol=0.01)
    assert np.allclose(traces, rec_car.get_traces() + np.mean(traces, axis=1, keepdims=True), atol=0.01)
    assert not np.all(rec_sin.get_traces()[0])
    assert np.allclose(rec_sin.get_traces()[:, 1], traces[:, 1] - traces[:, 0])

    assert np.allclose(traces[:, 0], rec_local_car.get_traces()[:, 0] + np.median(traces[:, [2, 3]], axis=1), atol=0.01)
    assert np.allclose(traces[:, 1], rec_local_car.get_traces()[:, 1] + np.median(traces[:, [3]], axis=1), atol=0.01)


if __name__ == "__main__":
    test_common_reference()
