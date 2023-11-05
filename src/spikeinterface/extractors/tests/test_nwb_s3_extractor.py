from pathlib import Path
import pickle

import pytest
import numpy as np
import h5py

from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.extractors import NwbRecordingExtractor, NwbSortingExtractor

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "extractors"
else:
    cache_folder = Path("cache_folder") / "extractors"


@pytest.mark.ros3_test
@pytest.mark.streaming_extractors
@pytest.mark.skipif("ros3" not in h5py.registered_drivers(), reason="ROS3 driver not installed")
def test_recording_s3_nwb_ros3(tmp_path):
    file_path = (
        "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc"
    )
    rec = NwbRecordingExtractor(file_path, stream_mode="ros3")

    start_frame = 0
    end_frame = 300
    num_frames = end_frame - start_frame

    num_seg = rec.get_num_segments()
    num_chans = rec.get_num_channels()
    dtype = rec.get_dtype()

    for segment_index in range(num_seg):
        num_samples = rec.get_num_samples(segment_index=segment_index)

        full_traces = rec.get_traces(segment_index=segment_index, start_frame=start_frame, end_frame=end_frame)
        assert full_traces.shape == (num_frames, num_chans)
        assert full_traces.dtype == dtype

    if rec.has_scaled():
        trace_scaled = rec.get_traces(segment_index=segment_index, return_scaled=True, end_frame=2)
        assert trace_scaled.dtype == "float32"

    tmp_file = tmp_path / "test_ros3_recording.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(rec, f)

    with open(tmp_file, "rb") as f:
        reloaded_recording = pickle.load(f)

    check_recordings_equal(rec, reloaded_recording)


@pytest.mark.streaming_extractors
def test_recording_s3_nwb_fsspec(tmp_path):
    file_path = (
        "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc"
    )
    rec = NwbRecordingExtractor(file_path, stream_mode="fsspec", stream_cache_path=cache_folder)

    start_frame = 0
    end_frame = 300
    num_frames = end_frame - start_frame

    num_seg = rec.get_num_segments()
    num_chans = rec.get_num_channels()
    dtype = rec.get_dtype()

    for segment_index in range(num_seg):
        num_samples = rec.get_num_samples(segment_index=segment_index)

        full_traces = rec.get_traces(segment_index=segment_index, start_frame=start_frame, end_frame=end_frame)
        assert full_traces.shape == (num_frames, num_chans)
        assert full_traces.dtype == dtype

    if rec.has_scaled():
        trace_scaled = rec.get_traces(segment_index=segment_index, return_scaled=True, end_frame=2)
        assert trace_scaled.dtype == "float32"

    tmp_file = tmp_path / "test_fsspec_recording.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(rec, f)

    with open(tmp_file, "rb") as f:
        reloaded_recording = pickle.load(f)

    check_recordings_equal(rec, reloaded_recording)


@pytest.mark.ros3_test
@pytest.mark.streaming_extractors
@pytest.mark.skipif("ros3" not in h5py.registered_drivers(), reason="ROS3 driver not installed")
def test_sorting_s3_nwb_ros3(tmp_path):
    file_path = "https://dandiarchive.s3.amazonaws.com/blobs/84b/aa4/84baa446-cf19-43e8-bdeb-fc804852279b"
    # we provide the 'sampling_frequency' because the NWB file does not the electrical series
    sort = NwbSortingExtractor(file_path, sampling_frequency=30000, stream_mode="ros3")

    start_frame = 0
    end_frame = 300
    num_frames = end_frame - start_frame

    num_seg = sort.get_num_segments()
    num_units = len(sort.unit_ids)

    for segment_index in range(num_seg):
        for unit in sort.unit_ids:
            spike_train = sort.get_unit_spike_train(unit_id=unit, segment_index=segment_index)
            assert len(spike_train) > 0
            assert spike_train.dtype == "int64"
            assert np.all(spike_train >= 0)

    tmp_file = tmp_path / "test_ros3_sorting.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(sort, f)

    with open(tmp_file, "rb") as f:
        reloaded_sorting = pickle.load(f)

    check_sortings_equal(reloaded_sorting, sort)


@pytest.mark.streaming_extractors
def test_sorting_s3_nwb_fsspec(tmp_path):
    file_path = "https://dandiarchive.s3.amazonaws.com/blobs/84b/aa4/84baa446-cf19-43e8-bdeb-fc804852279b"
    # we provide the 'sampling_frequency' because the NWB file does not the electrical series
    sort = NwbSortingExtractor(
        file_path, sampling_frequency=30000, stream_mode="fsspec", stream_cache_path=cache_folder
    )

    start_frame = 0
    end_frame = 300
    num_frames = end_frame - start_frame

    num_seg = sort.get_num_segments()
    num_units = len(sort.unit_ids)

    for segment_index in range(num_seg):
        for unit in sort.unit_ids:
            spike_train = sort.get_unit_spike_train(unit_id=unit, segment_index=segment_index)
            assert len(spike_train) > 0
            assert spike_train.dtype == "int64"
            assert np.all(spike_train >= 0)

    tmp_file = tmp_path / "test_fsspec_sorting.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(sort, f)

    with open(tmp_file, "rb") as f:
        reloaded_sorting = pickle.load(f)

    check_sortings_equal(reloaded_sorting, sort)


if __name__ == "__main__":
    test_recording_s3_nwb_ros3()
    test_recording_s3_nwb_fsspec()
    test_sorting_s3_nwb_ros3()
    test_sorting_s3_nwb_fsspec()
