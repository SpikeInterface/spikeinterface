

from spikeinterface.core.tests.testing_tools import generate_recording

from spikeinterface.core.job_tools import divide_into_chunks, ensure_n_jobs, ensure_chunk_size, ChunkRecordingExecutor



def test_devide_into_chunks():
    chunks =  devide_into_chunks(10, 5)
    assert len(chunks) == 2
    chunks =  devide_into_chunks(11, 5)
    assert len(chunks) == 3
    assert chunks[0] == (0, 5)
    assert chunks[1] == (5, 10)
    assert chunks[2] == (10, 11)


def test_ensure_n_jobs():
    recording = generate_recording()
    
    n_jobs = ensure_n_jobs(recording)
    assert n_jobs == 1

    n_jobs = ensure_n_jobs(recording, n_jobs=0)
    assert n_jobs == 1

    n_jobs = ensure_n_jobs(recording, n_jobs=1)
    assert n_jobs == 1
    
    # not dumpable force n_jobs=1
    n_jobs = ensure_n_jobs(recording, n_jobs=-1)
    assert n_jobs  == 1

    # dumpable
    n_jobs = ensure_n_jobs(recording.save(), n_jobs=-1)
    assert n_jobs  > 1
    

def test_ensure_chunk_size():
    recording = generate_recording(num_channels = 2)
    dtype = recording.get_dtype()
    assert dtype == 'float32'
    # make dumpable
    recording = recording.save()
    
    chunk_size = ensure_chunk_size(recording, total_memory="512M", chunk_size=None, chunk_memory=None, n_jobs=2)
    assert chunk_size == 32000000

    chunk_size = ensure_chunk_size(recording, chunk_memory="256M")
    assert chunk_size == 32000000
    
    chunk_size = ensure_chunk_size(recording, chunk_memory="1k")
    assert chunk_size == 125

    chunk_size = ensure_chunk_size(recording, chunk_memory="1G")
    assert chunk_size == 125000000
    
def test_ChunkRecordingExecutor():
    recording = generate_recording(num_channels = 2)
    # make dumpable
    recording =recording.save()
    
    def func(segment_index, start_frame, end_frame, worker_ctx):
        import os, time
        # print('func', segment_index, start_frame, end_frame, worker_ctx, os.getpid())
        time.sleep(0.010)
        #~ time.sleep(1.0)
        return os.getpid()
        
    
    def init_func(arg1, arg2, arg3):
        worker_ctx = {}
        worker_ctx['arg1'] = arg1
        worker_ctx['arg2'] = arg2
        worker_ctx['arg3'] = arg3
        return worker_ctx
    
    init_args = 'a', 120, 'yep'
    
    # no chunk
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, 
            verbose=True, progress_bar=False,
            n_jobs=1, chunk_size=None)
    processor.run()
    
    # chunk + loop
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, 
            verbose=True, progress_bar=False,
            n_jobs=1, chunk_memory="500k")
    processor.run()
    
    # chunk + parralel
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, 
            verbose=True, progress_bar=True,
            n_jobs=2, total_memory="200k",
            job_name='job_name')
    processor.run()

    
    
if __name__ == '__main__':
    #~ test_devide_into_chunks()
    #~ test_ensure_n_jobs()
    #~ test_ensure_chunk_size()
    test_ChunkRecordingExecutor()
    
    