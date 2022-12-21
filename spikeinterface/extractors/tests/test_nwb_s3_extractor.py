import pytest
import numpy as np
import h5py

from spikeinterface.extractors import NwbRecordingExtractor


@pytest.mark.skipif("ros3" not in h5py.registered_drivers(), reason="ROS3 driver not installed")
def test_s3_nwb_ros3():
    file_path = "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc"
    rec = NwbRecordingExtractor(file_path, stream_mode="ros3")
    
    start_frame = 0
    end_frame = 300
    num_frames = end_frame - start_frame

    num_seg = rec.get_num_segments()
    num_chans = rec.get_num_channels()
    dtype = rec.get_dtype()

    for segment_index in range(num_seg):
        num_samples = rec.get_num_samples(segment_index=segment_index)

        full_traces = rec.get_traces(segment_index=segment_index, start_frame=start_frame,
                                     end_frame=end_frame)
        assert full_traces.shape == (num_frames, num_chans)
        assert full_traces.dtype == dtype

    if rec.has_scaled():
        trace_scaled = rec.get_traces(segment_index=segment_index, return_scaled=True, end_frame=2)
        assert trace_scaled.dtype == 'float32'


def test_s3_nwb_fsspec():
    file_path = "https://dandi-api-staging-dandisets.s3.amazonaws.com/blobs/5f4/b7a/5f4b7a1f-7b95-4ad8-9579-4df6025371cc"
    rec = NwbRecordingExtractor(file_path, stream_mode="fsspec")
    
    start_frame = 0
    end_frame = 300
    num_frames = end_frame - start_frame

    num_seg = rec.get_num_segments()
    num_chans = rec.get_num_channels()
    dtype = rec.get_dtype()

    for segment_index in range(num_seg):
        num_samples = rec.get_num_samples(segment_index=segment_index)

        full_traces = rec.get_traces(segment_index=segment_index, start_frame=start_frame,
                                     end_frame=end_frame)
        assert full_traces.shape == (num_frames, num_chans)
        assert full_traces.dtype == dtype

    if rec.has_scaled():
        trace_scaled = rec.get_traces(segment_index=segment_index, return_scaled=True, end_frame=2)
        assert trace_scaled.dtype == 'float32'
