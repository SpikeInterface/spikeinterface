

from spikeinterface.core.tests.testing_tools import generate_recording

from spikeinterface.core.core_tools import write_binary_recording, write_memory_recording

def test_write_binary_recording():
    # 2 segments
    recording = generate_recording(num_channels = 2, durations = [10.325, 3.5])
    # make dumpable
    recording =recording.save()
    
    # write with loop
    write_binary_recording(recording, files_path=['binary01.raw', 'binary02.raw'], dtype=None,
            verbose=True, n_jobs=1)

    write_binary_recording(recording, files_path=['binary01.raw', 'binary02.raw'], dtype=None,
            verbose=True, n_jobs=1, chunk_memory='100k', progress_bar=True)


    # write parrallel
    write_binary_recording(recording, files_path=['binary01.raw', 'binary02.raw'], dtype=None,
            verbose=False, n_jobs=2, chunk_memory='100k')
    
    # write parrallel
    write_binary_recording(recording, files_path=['binary01.raw', 'binary02.raw'], dtype=None,
            verbose=False, n_jobs=2, total_memory='200k',  progress_bar=True)
    
def test_write_memory_recording():
    # 2 segments
    recording = generate_recording(num_channels = 2, durations = [10.325, 3.5])
    # make dumpable
    recording =recording.save()

    # write with loop
    #~ write_memory_recording(recording, dtype=None, verbose=True, n_jobs=1)

    #~ write_memory_recording(recording, dtype=None,
            #~ verbose=True, n_jobs=1, chunk_memory='100k', progress_bar=True)

    # write parrallel
    write_memory_recording(recording, dtype=None,
            verbose=False, n_jobs=2, chunk_memory='100k')

    # write parrallel
    write_memory_recording(recording, dtype=None,
            verbose=False, n_jobs=2, total_memory='200k',  progress_bar=True)


    
    
if __name__ == '__main__':
    #~ test_write_binary_recording()
    
    test_write_memory_recording()
    
