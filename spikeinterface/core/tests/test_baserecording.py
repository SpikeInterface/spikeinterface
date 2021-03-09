"""
test for BaseRecording are done with BinaryRecordingExtractor.
but check only for BaseRecording general methods.
"""
import shutil
from pathlib import Path
import pytest
import numpy as np

#~ import probeinterface as pi
from probeinterface import Probe

from spikeinterface.core import BinaryRecordingExtractor, load_extractor
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
        np.memmap(files_path[i], dtype=dtype, mode='w+', shape=(num_samples, num_chan))

    rec = BinaryRecordingExtractor(files_path, sampling_frequency, num_chan, dtype)
    print(rec)
    
    assert rec.get_num_segments() == 2
    assert rec.get_num_channels() == 3
    
    assert np.all(rec.ids_to_indices([0,1,2]) == [0, 1, 2])
    assert np.all(rec.ids_to_indices([0,1,2], prefer_slice=True) == slice(0,3, None))
    
    #~ # annotations / properties
    #~ rec.annotate(yep='yop')
    #~ assert rec.get_annotation('yep') == 'yop'
    
    #~ rec.set_property('quality', [1., 3.3, np.nan])
    #~ values = rec.get_property('quality')
    #~ assert np.all(values[:2] == [1., 3.3, ])
    
    #~ # dump/load dict
    #~ d = rec.to_dict()
    #~ rec2 = BaseExtractor.from_dict(d)
    #~ rec3 = load_extractor(d)
    
    #~ # dump/load json
    #~ rec.dump_to_json('test_BaseRecording.json')
    #~ rec2 = BaseExtractor.load('test_BaseRecording.json')
    #~ rec3 = load_extractor('test_BaseRecording.json')
    
    #~ # dump/load pickle
    #~ rec.dump_to_pickle('test_BaseRecording.pkl')
    #~ rec2 = BaseExtractor.load('test_BaseRecording.pkl')
    #~ rec3 = load_extractor('test_BaseRecording.pkl')
    
    #~ # cache
    #~ cache_folder = './my_cache_folder'
    #~ rec.set_cache_folder(cache_folder)
    #~ rec.cache(name='simple_recording')
    #~ rec2 = BaseExtractor.load_from_cache(cache_folder, 'simple_recording')
    #~ # but also possible
    #~ rec3 = BaseExtractor.load('./my_cache_folder/simple_recording')
    
    #~ # cache joblib several jobs
    #~ rec.cache(name='simple_recording_2', chunk_size=10, n_jobs=4)
    
    # set/get Probe only 2 channels
    probe = Probe(ndim=2)
    positions = [[0., 0.], [0., 20.], [0, 30.]]
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices([2, -1, 0])
    
    rec = rec.set_probe(probe)
    positions2 = rec.get_channel_locations()
    print(positions2)
    
    
    
    
    

    
    
    
    

    

if __name__ == '__main__':
    _clean_all()
    test_BaseRecording()

