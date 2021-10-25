import shutil
from pathlib import Path

import pytest
import numpy as np

import probeinterface as pi

from spikeinterface.core import ChannelSliceRecording, BinaryRecordingExtractor


def _clean_all():
    cache_folder = './my_cache_folder'
    if Path(cache_folder).exists():
        shutil.rmtree(cache_folder)


def test_ChannelSliceRecording():
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = 'int16'

    file_paths = [f'test_BinaryRecordingExtractor_{i}.raw' for i in range(num_seg)]
    for i in range(num_seg):
        traces = np.memmap(file_paths[i], dtype=dtype, mode='w+', shape=(num_samples, num_chan))
        traces[:] = np.arange(3)[None, :]
    rec = BinaryRecordingExtractor(file_paths, sampling_frequency, num_chan, dtype)

    # keep original ids
    rec_sliced = ChannelSliceRecording(rec, channel_ids=[0, 2])
    assert np.all(rec_sliced.get_channel_ids() == [0, 2])
    traces = rec_sliced.get_traces(segment_index=1)
    assert traces.shape[1] == 2
    traces = rec_sliced.get_traces(segment_index=1, channel_ids=[0, 2])
    assert traces.shape[1] == 2
    traces = rec_sliced.get_traces(segment_index=1, channel_ids=[2, 0])
    assert traces.shape[1] == 2
    
    assert np.allclose(rec_sliced.get_times(0), rec.get_times(0))
    
    # with channel ids renaming
    rec_sliced2 = ChannelSliceRecording(rec, channel_ids=[0, 2], renamed_channel_ids=[3, 4])
    assert np.all(rec_sliced2.get_channel_ids() == [3, 4])
    traces = rec_sliced2.get_traces(segment_index=1)
    assert traces.shape[1] == 2
    assert np.all(traces[:, 0] == 0)
    assert np.all(traces[:, 1] == 2)

    traces = rec_sliced2.get_traces(segment_index=1, channel_ids=[4, 3])
    assert traces.shape[1] == 2
    assert np.all(traces[:, 0] == 2)
    assert np.all(traces[:, 1] == 0)

    # with probe and after save()
    probe = pi.generate_linear_probe(num_elec=num_chan)
    probe.set_device_channel_indices(np.arange(num_chan))
    rec_p = rec.set_probe(probe)
    rec_sliced3 = ChannelSliceRecording(rec_p, channel_ids=[0, 2], renamed_channel_ids=[3, 4])
    probe3 = rec_sliced3.get_probe()
    locations3 = probe3.contact_positions
    cache_folder = Path('./my_cache_folder')
    folder = cache_folder / 'sliced_recording'
    rec_saved = rec_sliced3.save(folder=folder, chunk_size=10, n_jobs=2)
    probe = rec_saved.get_probe()
    assert np.array_equal(locations3, rec_saved.get_channel_locations())
    traces3 = rec_saved.get_traces(segment_index=0)
    assert np.all(traces3[:, 0] == 0)
    assert np.all(traces3[:, 1] == 2)


if __name__ == '__main__':
    _clean_all()
    test_ChannelSliceRecording()
