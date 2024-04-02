import operator
import pytest

from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import common_reference

import numpy as np
from tqdm import trange


def _generate_test_recording():
    recording = generate_recording(durations=[1.0], num_channels=4)
    recording = recording.channel_slice(recording.channel_ids, np.array(["a", "b", "c", "d"]))
    return recording


@pytest.fixture
def recording():
    return _generate_test_recording()


def test_common_reference(recording):
    # Test simple case
    rec_cmr = common_reference(recording, reference="global", operator="median")
    rec_car = common_reference(recording, reference="global", operator="average")
    rec_sin = common_reference(recording, reference="single", ref_channel_ids=["a"])
    rec_local_car = common_reference(recording, reference="local", local_radius=(20, 65), operator="median")

    traces = recording.get_traces()
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


def test_common_reference_channel_slicing(recording):
    recording_cmr = common_reference(recording, reference="global", operator="median")
    recording_car = common_reference(recording, reference="global", operator="average")
    recording_single_reference = common_reference(recording, reference="single", ref_channel_ids=["b"])
    recording_local_car = common_reference(recording, reference="local", local_radius=(20, 65), operator="median")

    channel_ids = ["b", "d"]
    indices = recording.ids_to_indices(channel_ids)

    all_channel_ids = recording.channel_ids
    all_indices = recording.ids_to_indices(all_channel_ids)

    original_traces = recording.get_traces()

    cmr_trace = recording_cmr.get_traces(channel_ids=channel_ids)
    expected_trace = original_traces[:, indices] - np.median(original_traces, axis=1, keepdims=True)
    assert np.allclose(cmr_trace, expected_trace, atol=0.01)

    car_trace = recording_car.get_traces(channel_ids=channel_ids)
    expected_trace = original_traces[:, indices] - np.mean(original_traces, axis=1, keepdims=True)
    assert np.allclose(car_trace, expected_trace, atol=0.01)

    single_reference_trace = recording_single_reference.get_traces(channel_ids=channel_ids)
    single_reference_index = recording.ids_to_indices(["b"])
    expected_trace = original_traces[:, indices] - original_traces[:, single_reference_index]

    assert np.allclose(single_reference_trace, expected_trace, atol=0.01)

    local_trace = recording_local_car.get_traces(channel_ids=all_channel_ids)
    local_trace_sub = recording_local_car.get_traces(channel_ids=channel_ids)

    assert np.all(local_trace[:, indices] == local_trace_sub)

    # test segment slicing

    start_frame = 0
    end_frame = 10

    recording_segment_cmr = recording_cmr._recording_segments[0]
    traces_cmr_all = recording_segment_cmr.get_traces(
        start_frame=start_frame, end_frame=end_frame, channel_indices=all_indices
    )
    traces_cmr_sub = recording_segment_cmr.get_traces(
        start_frame=start_frame, end_frame=end_frame, channel_indices=indices
    )

    assert np.all(traces_cmr_all[:, indices] == traces_cmr_sub)

    recording_segment_car = recording_car._recording_segments[0]
    traces_car_all = recording_segment_car.get_traces(
        start_frame=start_frame, end_frame=end_frame, channel_indices=all_indices
    )
    traces_car_sub = recording_segment_car.get_traces(
        start_frame=start_frame, end_frame=end_frame, channel_indices=indices
    )

    assert np.all(traces_car_all[:, indices] == traces_car_sub)

    recording_segment_local = recording_local_car._recording_segments[0]
    traces_local_all = recording_segment_local.get_traces(
        start_frame=start_frame, end_frame=end_frame, channel_indices=all_indices
    )
    traces_local_sub = recording_segment_local.get_traces(
        start_frame=start_frame, end_frame=end_frame, channel_indices=indices
    )

    assert np.all(traces_local_all[:, indices] == traces_local_sub)


def test_common_reference_groups(recording):
    original_traces = recording.get_traces()
    groups = [["a", "c", "d"], ["b"]]

    # median - global
    recording_cmr = common_reference(recording, reference="global", operator="median", groups=groups)
    traces = recording_cmr.get_traces(channel_ids=["c", "b", "d"])
    # c/d
    assert np.allclose(
        traces[:, 0], original_traces[:, 2] - np.median(original_traces[:, [0, 2, 3]], axis=1), atol=0.01
    )
    assert np.allclose(
        traces[:, 2], original_traces[:, 3] - np.median(original_traces[:, [0, 2, 3]], axis=1), atol=0.01
    )
    # b will be all zeros
    assert np.allclose(traces[:, 1], 0)

    traces = recording_cmr.get_traces(channel_ids=["a"])
    assert np.allclose(
        traces[:, 0], original_traces[:, 0] - np.median(original_traces[:, [0, 2, 3]], axis=1), atol=0.01
    )

    # single
    groups = [["a", "c"], ["b", "d"]]
    recording_single = common_reference(recording, reference="single", ref_channel_ids=["a", "b"], groups=groups)
    traces = recording_single.get_traces(channel_ids=["c", "b", "a", "d"])
    # c - referenced to a
    assert np.allclose(traces[:, 0], original_traces[:, 2] - original_traces[:, 0], atol=0.01)
    # b - all zeros
    assert np.allclose(traces[:, 1], 0)
    # a - all zeros
    assert np.allclose(traces[:, 2], 0)
    # d - referenced to b
    assert np.allclose(traces[:, 3], original_traces[:, 3] - original_traces[:, 1], atol=0.01)

    # TODO: fix this!
    traces = recording_single.get_traces(channel_ids=["d", "b"])
    # d - referenced to b
    assert np.allclose(traces[:, 0], original_traces[:, 3] - original_traces[:, 1], atol=0.01)
    # b - all zeros
    assert np.allclose(traces[:, 1], 0)


if __name__ == "__main__":
    recording = _generate_test_recording()
    test_common_reference(recording)
    test_common_reference_channel_slicing(recording)
    test_common_reference_groups(recording)
