from pathlib import Path
import pickle
from tabnanny import check

import pytest
import numpy as np
import h5py

from spikeinterface import load_extractor
from spikeinterface.core.testing import check_recordings_equal
from spikeinterface.core.testing import check_recordings_equal, check_sortings_equal
from spikeinterface.extractors import NwbRecordingExtractor, NwbSortingExtractor


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
@pytest.mark.parametrize("cache", [True, False])  # Test with and without cache
def test_recording_s3_nwb_fsspec(tmp_path, cache):
    file_path = (
        "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc"
    )

    # Instantiate NwbRecordingExtractor with the cache parameter
    rec = NwbRecordingExtractor(
        file_path, stream_mode="fsspec", cache=cache, stream_cache_path=tmp_path if cache else None
    )

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


@pytest.mark.streaming_extractors
def test_recording_s3_nwb_remfile():
    file_path = (
        "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc"
    )
    rec = NwbRecordingExtractor(file_path, stream_mode="remfile")

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


@pytest.mark.streaming_extractors
def test_recording_s3_nwb_remfile_file_like(tmp_path):
    import remfile

    file_path = (
        "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc"
    )
    file = remfile.File(file_path)
    rec = NwbRecordingExtractor(file=file)

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

    # test pickling
    with open(tmp_path / "rec.pkl", "wb") as f:
        pickle.dump(rec, f)
    with open(tmp_path / "rec.pkl", "rb") as f:
        rec2 = pickle.load(f)
    check_recordings_equal(rec, rec2)


@pytest.mark.streaming_extractors
@pytest.mark.skipif("ros3" not in h5py.registered_drivers(), reason="ROS3 driver not installed")
def test_sorting_s3_nwb_ros3(tmp_path):
    file_path = "https://dandiarchive.s3.amazonaws.com/blobs/84b/aa4/84baa446-cf19-43e8-bdeb-fc804852279b"
    # we provide the 'sampling_frequency' because the NWB file does not the electrical series
    sort = NwbSortingExtractor(file_path, sampling_frequency=30000, stream_mode="ros3", t_start=0)

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
@pytest.mark.parametrize("cache", [True, False])  # Test with and without cache
def test_sorting_s3_nwb_fsspec(tmp_path, cache):
    file_path = "https://dandiarchive.s3.amazonaws.com/blobs/84b/aa4/84baa446-cf19-43e8-bdeb-fc804852279b"
    # We provide the 'sampling_frequency' because the NWB file does not have the electrical series
    sorting = NwbSortingExtractor(
        file_path,
        sampling_frequency=30000.0,
        stream_mode="fsspec",
        cache=cache,
        stream_cache_path=tmp_path if cache else None,
        t_start=0,
    )

    num_seg = sorting.get_num_segments()
    assert num_seg == 1
    num_units = len(sorting.unit_ids)
    assert num_units == 64

    for segment_index in range(num_seg):
        for unit in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id=unit, segment_index=segment_index)
            assert len(spike_train) > 0
            assert spike_train.dtype == "int64"
            assert np.all(spike_train >= 0)

    tmp_file = tmp_path / "test_fsspec_sorting.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(sorting, f)

    with open(tmp_file, "rb") as f:
        reloaded_sorting = pickle.load(f)

    check_sortings_equal(reloaded_sorting, sorting)


@pytest.mark.streaming_extractors
def test_sorting_s3_nwb_remfile(tmp_path):
    file_path = "https://dandiarchive.s3.amazonaws.com/blobs/84b/aa4/84baa446-cf19-43e8-bdeb-fc804852279b"
    # We provide the 'sampling_frequency' because the NWB file does not have the electrical series
    sorting = NwbSortingExtractor(
        file_path,
        sampling_frequency=30000.0,
        stream_mode="remfile",
        t_start=0,
    )

    num_seg = sorting.get_num_segments()
    assert num_seg == 1
    num_units = len(sorting.unit_ids)
    assert num_units == 64

    for segment_index in range(num_seg):
        for unit in sorting.unit_ids:
            spike_train = sorting.get_unit_spike_train(unit_id=unit, segment_index=segment_index)
            assert len(spike_train) > 0
            assert spike_train.dtype == "int64"
            assert np.all(spike_train >= 0)

    tmp_file = tmp_path / "test_remfile_sorting.pkl"
    with open(tmp_file, "wb") as f:
        pickle.dump(sorting, f)

    with open(tmp_file, "rb") as f:
        reloaded_sorting = pickle.load(f)

    check_sortings_equal(reloaded_sorting, sorting)


@pytest.mark.streaming_extractors
def test_sorting_s3_nwb_zarr(tmp_path):
    file_path = (
        "s3://aind-open-data/ecephys_625749_2022-08-03_15-15-06_nwb_2023-05-16_16-34-55/"
        "ecephys_625749_2022-08-03_15-15-06_nwb/"
        "ecephys_625749_2022-08-03_15-15-06_experiment1_recording1.nwb.zarr/"
    )
    # We provide the 'sampling_frequency' because the NWB file obly has LFP electrical series
    sorting = NwbSortingExtractor(
        file_path,
        sampling_frequency=30000.0,
        stream_mode="zarr",
        storage_options={"anon": True},
        t_start=0,
        load_unit_properties=False,
    )

    num_seg = sorting.get_num_segments()
    assert num_seg == 1
    num_units = len(sorting.unit_ids)
    assert num_units == 456

    # This is too slow for testing
    # for segment_index in range(num_seg):
    #     for unit in sorting.unit_ids:
    #         spike_train = sorting.get_unit_spike_train(unit_id=unit, segment_index=segment_index)
    #         assert len(spike_train) > 0
    #         assert spike_train.dtype == "int64"
    #         assert np.all(spike_train >= 0)

    # with this mode, the object is not serializable
    assert not sorting.check_serializability("json")
    assert not sorting.check_serializability("pickle")

    # test to/from dict
    sorting_loaded = load_extractor(sorting.to_dict())
    check_sortings_equal(sorting, sorting_loaded)


if __name__ == "__main__":
    tmp_path = Path("tmp")
    if tmp_path.is_dir():
        import shutil

        shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    test_recording_s3_nwb_fsspec(tmp_path, cache=True)
