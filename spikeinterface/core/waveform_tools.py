
import numpy as np

from .job_tools import ChunkRecordingExecutor, ensure_n_jobs, _shared_job_kwargs_doc

def allocate_waveforms(recording, spikes, unit_ids, nbefore, nafter, mode='memmap', folder=None, dtype=None, sparsity_mask=None):
    """
    
    
    sparsity_mask:
        If not None must
        shape=(len(unit_ids), len(channel_ids)
        dtype='bool'
        
    
    """
    num_chans = recording.get_num_channels()
    nsamples = nbefore + nafter
    
    if mode =='memmap':
        # prepare memmap
        wfs_arrays = {}
        for unit_ind, unit_id in enumerate(unit_ids):
            n_spikes = np.sum(spikes['unit_ind'] == unit_ind)
            shape = (n_spikes, nsamples, num_chans)
            filename = folder / f'waveforms_{unit_id}.npy'
            arr = np.lib.format.open_memmap(filename, mode='w+', dtype=dtype, shape=shape)
            wfs_arrays[unit_id] = arr
        return wfs_arrays
    elif mode =='memory':
        raise NotImplementedError()
    else:
        raise ValueError('allocate_waveforms bad mode')

def distribute_waveform_to_buffers(recording, spikes, unit_ids, wfs_arrays, nbefore, nafter, return_scaled, **job_kwargs):
    n_jobs = ensure_n_jobs(recording, job_kwargs.get('n_jobs', None))

    # and run
    func = _waveform_extractor_chunk
    init_func = _init_worker_waveform_extractor
    if n_jobs == 1:
        init_args = (recording, )
    else:
        init_args = (recording.to_dict(), )
    init_args = init_args + (unit_ids, spikes, wfs_arrays, nbefore, nafter, return_scaled)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, job_name='extract waveforms',
                                       **job_kwargs)
    processor.run()




# used by WaveformExtractor + ChunkRecordingExecutor
def _init_worker_waveform_extractor(recording, unit_ids, spikes, wfs_arrays, nbefore, nafter, return_scaled):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording
    
    worker_ctx['unit_ids'] = unit_ids
    worker_ctx['spikes'] = spikes
    worker_ctx['wfs_arrays'] = wfs_arrays
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter
    worker_ctx['return_scaled'] = return_scaled
    
    inds_by_unit = {}
    for unit_ind, unit_id in enumerate(unit_ids):
        inds,  = np.nonzero(spikes['unit_ind'] == unit_ind)
        inds_by_unit[unit_id] = inds
    worker_ctx['inds_by_unit'] = inds_by_unit

    return worker_ctx


# used by WaveformExtractor + ChunkRecordingExecutor
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
            #~ print('l0', l0, 'l1', l1, 'unit_ind', unit_ind, 'in_chunk_pos', in_chunk_pos)
            for pos in in_chunk_pos:
                sample_ind = spikes[inds[pos]]['sample_ind']
                #~ print('sample_ind', sample_ind, 'segment_index', segment_index, 'seg_size', seg_size, 'start', start, 'end', end)
                wfs[pos,: , :] = traces[sample_ind - start - nbefore:sample_ind - start + nafter, :]

