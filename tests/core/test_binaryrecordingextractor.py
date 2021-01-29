import pytest
import numpy as np
from pathlib import Path

from spikeinterface.core import BinaryRecordingExtractor


def test_BinaryRecordingExtractor():
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = 'int16'
    
    files_path = [f'test_BinaryRecordingExtractor_{i}.raw' for i in range(num_seg)]
    for i in range(num_seg):
        np.memmap(files_path[i], dtype=dtype, mode='w+', shape=(num_samples, num_chan))
        
    rec = BinaryRecordingExtractor(files_path, sampling_frequency, num_chan, dtype)
    print(rec)
    
    files_path = [f'test_BinaryRecordingExtractor_copied_{i}.raw' for i in range(num_seg)]
    BinaryRecordingExtractor.write_recording(rec, files_path)
    
    assert Path('test_BinaryRecordingExtractor_copied_0.raw').is_file()
    


if __name__ == '__main__':
    test_BinaryRecordingExtractor()
