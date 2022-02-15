"""
This module contain low level function to retrieve snippet of traces aka "spike waveforms".

This is internally used by WaveformExtractor but also can be used as a sorting component.

It is a 2 steps way :
  1. allocate buffer (file or memory)
  2. extract and distribute snipet to buffer (in parallel optionaly)


"""
import platform

import numpy as np

from .job_tools import ChunkRecordingExecutor, ensure_n_jobs, _shared_job_kwargs_doc
from .core_tools import make_shared_array


def allocate_waveforms(recording, spikes, unit_ids, nbefore, nafter, mode='memmap', folder=None, dtype=None, sparsity_mask=None):
    """
    
    
    sparsity_mask:
        If not None must
        shape=(len(unit_ids), len(channel_ids)
        dtype='bool'
        
    
    """
    
    nsamples = nbefore + nafter
    
    dtype = np.dtype(dtype)
    if mode =='shared_memory':
        assert folder is None
    
    # prepare buffers
    wfs_arrays = {}
    wfs_arrays_info = {}
    for unit_ind, unit_id in enumerate(unit_ids):
        n_spikes = np.sum(spikes['unit_ind'] == unit_ind)
        if sparsity_mask is None:
            num_chans = recording.get_num_channels()
        else:
            num_chans = np.sum(sparsity_mask[unit_ind, :])
        shape = (n_spikes, nsamples, num_chans)
        
        if mode =='memmap':
            filename = str(folder / f'waveforms_{unit_id}.npy')
            arr = np.lib.format.open_memmap(filename, mode='w+', dtype=dtype, shape=shape)
            wfs_arrays[unit_id] = arr
            wfs_arrays_info[unit_id] = filename
        elif mode =='shared_memory':
            arr, shm = make_shared_array(shape, dtype)
            wfs_arrays[unit_id] = arr
            wfs_arrays_info[unit_id] = (shm, shm.name, dtype.str, shape)
        else:
            raise ValueError('allocate_waveforms bad mode')

    return wfs_arrays, wfs_arrays_info
    

def distribute_waveforms_to_buffers(recording, spikes, unit_ids, wfs_arrays_info, nbefore, nafter, return_scaled, mode='memmap', sparsity_mask=None, **job_kwargs):
    n_jobs = ensure_n_jobs(recording, job_kwargs.get('n_jobs', None))

    inds_by_unit = {}
    for unit_ind, unit_id in enumerate(unit_ids):
        inds,  = np.nonzero(spikes['unit_ind'] == unit_ind)
        inds_by_unit[unit_id] = inds

    # and run
    func = _waveform_extractor_chunk
    init_func = _init_worker_waveform_extractor
    if n_jobs == 1:
        init_args = (recording, )
    else:
        init_args = (recording.to_dict(), )
    init_args = init_args + (unit_ids, spikes, wfs_arrays_info, nbefore, nafter, return_scaled, inds_by_unit, mode, sparsity_mask)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, job_name=f'extract waveforms {mode}',
                                       **job_kwargs)
    processor.run()


# used by ChunkRecordingExecutor
def _init_worker_waveform_extractor(recording, unit_ids, spikes, wfs_arrays_info, nbefore, nafter, return_scaled, inds_by_unit, mode, sparsity_mask):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording
    
    if mode == 'memmap':
        #~ if not platform.system().lower().startswith('linux'):
        # For OSX and windows : need to re open all npy files in r+ mode for each worker
        wfs_arrays = {}
        for unit_id, filename in wfs_arrays_info.items():
            wfs_arrays[unit_id] = np.load(filename, mmap_mode='r+')
    elif mode == 'shared_memory':
        
        from multiprocessing.shared_memory import SharedMemory
        wfs_arrays = {}
        shms = {}
        for unit_id, (sm, shm_name, dtype, shape) in wfs_arrays_info.items():
            shm = SharedMemory(shm_name)
            arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            wfs_arrays[unit_id] = arr
            # we need a reference to all sham otherwise we get segment fault!!!
            shms[unit_id] = shm
        worker_ctx['shms'] = shms

    worker_ctx['unit_ids'] = unit_ids
    worker_ctx['spikes'] = spikes
    worker_ctx['wfs_arrays'] = wfs_arrays
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter
    worker_ctx['return_scaled'] = return_scaled
    worker_ctx['inds_by_unit'] = inds_by_unit
    worker_ctx['sparsity_mask'] = sparsity_mask

    return worker_ctx


# used by ChunkRecordingExecutor
def _waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    unit_ids = worker_ctx['unit_ids']
    spikes = worker_ctx['spikes']
    wfs_arrays = worker_ctx['wfs_arrays']
    nbefore = worker_ctx['nbefore']
    nafter = worker_ctx['nafter']
    return_scaled = worker_ctx['return_scaled']
    inds_by_unit = worker_ctx['inds_by_unit']
    sparsity_mask = worker_ctx['sparsity_mask']

    seg_size = recording.get_num_samples(segment_index=segment_index)
    
    # take only spikes with the correct segment_ind
    # this is a slice so no copy!!
    s0 = np.searchsorted(spikes['segment_ind'], segment_index)
    s1 = np.searchsorted(spikes['segment_ind'], segment_index + 1)
    in_seg_spikes = spikes[s0:s1]
    
    # take only spikes in range [start_frame, end_frame]
    # this is a slice so no copy!!
    i0 = np.searchsorted(in_seg_spikes['sample_ind'], start_frame)
    i1 = np.searchsorted(in_seg_spikes['sample_ind'], end_frame)
    if i0 != i1:
        # protect from spikes on border :  spike_time<0 or spike_time>seg_size
        # useful only when max_spikes_per_unit is not None
        # waveform will not be extracted and a zeros will be left in the memmap file
        while (in_seg_spikes[i0]['sample_ind'] - nbefore) < 0 and (i0!=i1):
            i0 = i0 + 1
        while (in_seg_spikes[i1-1]['sample_ind'] + nafter) > seg_size and (i0!=i1):
            i1 = i1 - 1
    
    # slice in absolut in spikes vector
    l0 = i0 + s0
    l1 = i1 + s0
    
    if spikes.size > 0:
        start = spikes[l0]['sample_ind'] - nbefore
        end = spikes[l1-1]['sample_ind'] + nafter
        
        # load trace in memory
        traces = recording.get_traces(start_frame=start, end_frame=end, segment_index=segment_index,
                                      return_scaled=return_scaled)
        
        for unit_ind, unit_id in enumerate(unit_ids):
            wfs = wfs_arrays[unit_id]
            # find pos 
            inds = inds_by_unit[unit_id]
            in_chunk_pos,  = np.nonzero((inds >=l0) & (inds < l1))
            for pos in in_chunk_pos:
                sample_ind = spikes[inds[pos]]['sample_ind']
                wf = traces[sample_ind - start - nbefore:sample_ind - start + nafter, :]
                
                if sparsity_mask is None:
                    wfs[pos,: , :] = wf
                else:
                    wfs[pos,: , :] = wf[:, sparsity_mask[unit_ind]]

