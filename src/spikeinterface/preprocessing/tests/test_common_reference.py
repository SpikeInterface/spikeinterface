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

    # Test simple case
    rec_cmr = common_reference(rec, reference="global", operator="median")
    rec_car = common_reference(rec, reference="global", operator="average")
    rec_sin = common_reference(rec, reference="single", ref_channel_ids=["a"])
    rec_local_car = common_reference(rec, reference="local", local_radius=(20, 65), operator="median")

    traces = rec.get_traces()
    assert np.allclose(traces, rec_cmr.get_traces() + np.median(traces, axis=1, keepdims=True), atol=0.01)
    assert np.allclose(traces, rec_car.get_traces() + np.mean(traces, axis=1, keepdims=True), atol=0.01)
    assert not np.all(rec_sin.get_traces()[0])
    assert np.allclose(rec_sin.get_traces()[:, 1], traces[:, 1] - traces[:, 0])

    assert np.allclose(traces[:, 0], rec_local_car.get_traces()[:, 0] + np.median(traces[:, [2, 3]], axis=1), atol=0.01)
    assert np.allclose(traces[:, 1], rec_local_car.get_traces()[:, 1] + np.median(traces[:, [3]], axis=1), atol=0.01)

    # Saving tests
    rec_cmr.save(verbose=False)
    rec_car.save(verbose=False)
    rec_sin.save(verbose=False)
    rec_local_car.save(verbose=False)


def test_common_reference_channel_slicing():
    recording = generate_recording(durations=[1.0], num_channels=4)
    recording._main_ids = np.array(["a", "b", "c", "d"])

    recording_cmr = common_reference(recording, reference="global", operator="median")
    recording_car = common_reference(recording, reference="global", operator="average")
    recording_single_reference = common_reference(recording, reference="single", ref_channel_ids=["a"])

    channel_ids = ["a", "b"]
    indices = recording.ids_to_indices(["a", "b"])
    original_traces = recording.get_traces()

    cmr_trace = recording_cmr.get_traces(channel_ids=channel_ids)
    expected_trace = original_traces[:, indices] - np.median(original_traces, axis=1, keepdims=True)
    assert np.allclose(cmr_trace, expected_trace, atol=0.01)

    car_trace = recording_car.get_traces(channel_ids=channel_ids)
    expected_trace = original_traces[:, indices] - np.mean(original_traces, axis=1, keepdims=True)
    assert np.allclose(car_trace, expected_trace, atol=0.01)

    single_reference_trace = recording_single_reference.get_traces(channel_ids=channel_ids)
    single_reference_index = recording.ids_to_indices(["a"])
    expected_trace = original_traces[:, indices] - original_traces[:, single_reference_index]

    assert np.allclose(single_reference_trace, expected_trace, atol=0.01)


if __name__ == "__main__":
    test_common_reference()
