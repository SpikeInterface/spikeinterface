import pytest
import numpy as np

from spikeinterface.core import ChannelSliceRecording, BinaryRecordingExtractor


def test_ChannelSliceRecording():
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = 'int16'
    
    files_path = [f'test_BinaryRecordingExtractor_{i}.raw' for i in range(num_seg)]
    for i in range(num_seg):
        traces = np.memmap(files_path[i], dtype=dtype, mode='w+', shape=(num_samples, num_chan))
        traces[:] = np.arange(3)[None, :]
    rec = BinaryRecordingExtractor(files_path, sampling_frequency, num_chan, dtype)

    # keep original ids
    rec_sliced = ChannelSliceRecording(rec, channel_ids=[0, 2])
    assert np.all(rec_sliced.get_channel_ids() == [0, 2])
    traces = rec_sliced.get_traces(segment_index=1)
    assert traces.shape[1] == 2
    traces = rec_sliced.get_traces(segment_index=1, channel_ids=[0, 2])
    assert traces.shape[1] == 2
    traces = rec_sliced.get_traces(segment_index=1, channel_ids=[2, 0])
    assert traces.shape[1] == 2
    
    # with channel ids renaming
    rec_sliced2 = ChannelSliceRecording(rec, channel_ids=[0, 2], renamed_channel_ids=[3,4])
    assert np.all(rec_sliced2.get_channel_ids() == [3, 4])
    traces = rec_sliced2.get_traces(segment_index=1)
    assert traces.shape[1] == 2
    assert np.all(traces[:, 0] == 0)
    assert np.all(traces[:, 1] == 2)
    
    traces = rec_sliced2.get_traces(segment_index=1, channel_ids=[4, 3])
    assert traces.shape[1] == 2
    assert np.all(traces[:, 0] == 2)
    assert np.all(traces[:, 1] == 0)
    

if __name__ == '__main__':
    test_ChannelSliceRecording()

