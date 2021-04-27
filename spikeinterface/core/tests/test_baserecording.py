"""
test for BaseRecording are done with BinaryRecordingExtractor.
but check only for BaseRecording general methods.
"""
import shutil
from pathlib import Path
import pytest
import numpy as np


from probeinterface import Probe

from spikeinterface.core import BinaryRecordingExtractor, NumpyRecording, load_extractor
from spikeinterface.core.base import BaseExtractor

# file and folder created



def _clean_all():
    cache_folder = './my_cache_folder'
    if Path(cache_folder).exists():
        shutil.rmtree(cache_folder)
    
def setup_module():
    _clean_all()

def teardown_module():
    _clean_all()

def test_BaseRecording():
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = 'int16'
    
    files_path = [f'test_base_recording_{i}.raw' for i in range(num_seg)]
    for i in range(num_seg):
        a = np.memmap(files_path[i], dtype=dtype, mode='w+', shape=(num_samples, num_chan))
        a[:] = np.random.randn(*a.shape).astype(dtype)

    rec = BinaryRecordingExtractor(files_path, sampling_frequency, num_chan, dtype)
    print(rec)
    
    assert rec.get_num_segments() == 2
    assert rec.get_num_channels() == 3
    
    assert np.all(rec.ids_to_indices([0,1,2]) == [0, 1, 2])
    assert np.all(rec.ids_to_indices([0,1,2], prefer_slice=True) == slice(0,3, None))
    
    # annotations / properties
    rec.annotate(yep='yop')
    assert rec.get_annotation('yep') == 'yop'
    
    rec.set_property('quality', [1., 3.3, np.nan])
    values = rec.get_property('quality')
    assert np.all(values[:2] == [1., 3.3, ])
    
    # dump/load dict
    d = rec.to_dict()
    rec2 = BaseExtractor.from_dict(d)
    rec3 = load_extractor(d)
    
    # dump/load json
    rec.dump_to_json('test_BaseRecording.json')
    rec2 = BaseExtractor.load('test_BaseRecording.json')
    rec3 = load_extractor('test_BaseRecording.json')
    
    # dump/load pickle
    rec.dump_to_pickle('test_BaseRecording.pkl')
    rec2 = BaseExtractor.load('test_BaseRecording.pkl')
    rec3 = load_extractor('test_BaseRecording.pkl')
    
    # cache to binary
    cache_folder = Path('./my_cache_folder')
    folder = cache_folder / 'simple_recording'
    rec.save(format='binary', folder=folder)
    rec2 = BaseExtractor.load_from_folder(folder)
    assert 'quality' in rec2.get_property_keys()
    # but also possible
    rec3 = BaseExtractor.load('./my_cache_folder/simple_recording')
    
    # cache to memory
    rec4 = rec3.save(format='memory')
    
    traces4 = rec4.get_traces(segment_index=0)
    traces = rec.get_traces(segment_index=0)
    assert np.array_equal(traces4, traces)
    
    # cache joblib several jobs
    rec.save(name='simple_recording_2', chunk_size=10, n_jobs=4)
    
    # set/get Probe only 2 channels
    probe = Probe(ndim=2)
    positions = [[0., 0.], [0., 15.], [0, 30.]]
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices([2, -1, 0])
    probe.create_auto_shape()
    
    rec2 = rec.set_probe(probe, group_mode='by_shank')
    rec2 = rec.set_probe(probe, group_mode='by_probe')
    positions2 = rec2.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.], [0., 0.]])
    
    probe2 = rec2.get_probe()
    positions3 = probe2.contact_positions
    assert np.array_equal(positions2, positions3)
    

    # from probeinterface.plotting import plot_probe_group, plot_probe
    # import matplotlib.pyplot as plt
    # plot_probe(probe)
    # plot_probe(probe2)
    # plt.show()
    
    # test return_scale
    sampling_frequency = 30000
    traces = np.zeros((1000, 5), dtype='int16')
    rec_int16 = NumpyRecording([traces], sampling_frequency)
    assert rec_int16.get_dtype() == 'int16'
    print(rec_int16)
    traces_int16 = rec_int16.get_traces()
    assert traces_int16.dtype == 'int16'
    # return_scaled raise error when no gain_to_uV/offset_to_uV properties
    with pytest.raises(ValueError):
        traces_float32 = rec_int16.get_traces(return_scaled=True)
    rec_int16.set_property('gain_to_uV', [.195]*5)
    rec_int16.set_property('offset_to_uV', [0.]*5)
    traces_float32 = rec_int16.get_traces(return_scaled=True)
    assert traces_float32.dtype == 'float32'
    

if __name__ == '__main__':
    _clean_all()
    test_BaseRecording()

