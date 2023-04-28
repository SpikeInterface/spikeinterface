import platform
import pytest
from pathlib import Path

from spikeinterface.core import generate_recording

from spikeinterface.core.core_tools import write_binary_recording, write_memory_recording, recursive_path_modifier
from spikeinterface.core.binaryrecordingextractor import BinaryRecordingExtractor

try:
    from multiprocessing.shared_memory import SharedMemory

    HAVE_SHAREDMEMORY = True
except:
    HAVE_SHAREDMEMORY = False


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_write_binary_recording(tmp_path):
    # 2 segments
    from spikeinterface.core.generate import GeneratorRecording
    sampling_frequency = 30_000
    num_channels = 2
    dtype = "float32"
    recording = GeneratorRecording(durations=[10.325, 3.5], num_channels=num_channels, 
                                   sampling_frequency=sampling_frequency)

    # write with loop
    file_paths = [tmp_path / 'binary01.raw', tmp_path / 'binary02.raw']
    write_binary_recording(recording, file_paths=file_paths,
                           dtype=dtype, verbose=True, n_jobs=1)

    recorder_binary = BinaryRecordingExtractor(file_paths=file_paths, sampling_frequency=sampling_frequency, num_chan=num_channels, dtype=dtype)
    import numpy as np
    assert np.allclose(recorder_binary.get_traces(segment_index=0), recording.get_traces(segment_index=0))
    
    write_binary_recording(recording, file_paths=file_paths,
                           dtype=dtype, verbose=True, n_jobs=1, chunk_memory='100k', progress_bar=True)

    # write parrallel
    write_binary_recording(recording, file_paths=file_paths,
                           dtype=dtype, verbose=False, n_jobs=2, chunk_memory='100k')

    # write parrallel
    write_binary_recording(recording, file_paths=file_paths,
                           dtype=dtype, verbose=False, n_jobs=2, total_memory='200k', progress_bar=True)



def test_write_memory_recording():
    # 2 segments
    recording = generate_recording(num_channels=2, durations=[10.325, 3.5])
    # make dumpable
    recording = recording.save()

    # write with loop
    write_memory_recording(recording, dtype=None, verbose=True, n_jobs=1)

    write_memory_recording(recording, dtype=None,
                           verbose=True, n_jobs=1, chunk_memory='100k', progress_bar=True)

    if HAVE_SHAREDMEMORY and platform.system() != 'Windows':
        # write parrallel
        write_memory_recording(recording, dtype=None,
                               verbose=False, n_jobs=2, chunk_memory='100k')

        # write parrallel
        write_memory_recording(recording, dtype=None,
                               verbose=False, n_jobs=2, total_memory='200k', progress_bar=True)



def test_recursive_path_modifier():
    # this test nested depth 2 path modifier
    d = {
        'kwargs':{
            'path' : '/yep/path1',
            'recording': {
                'module': 'mock_module',
                'class': 'mock_class',
                'version': '1.2',
                'annotations': {},
                'kwargs': {
                    'path':'/yep/path2'
                },
            
            }
        }
    }

    d2  =recursive_path_modifier(d, lambda p: p.replace('/yep', '/yop'))
    assert d2['kwargs']['path'].startswith('/yop')
    assert d2['kwargs']['recording']['kwargs'] ['path'].startswith('/yop')


if __name__ == '__main__':
    # Create a temporary folder using the standard library
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        test_write_binary_recording(tmp_path)
        #test_write_memory_recording()
        #test_recursive_path_modifier()
