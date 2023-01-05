"""
This module contains low-level functions to extract snippets of traces (aka "spike waveforms").

This is internally used by WaveformExtractor, but can also be used as a sorting component.

It is a 2-step approach:
  1. allocate buffers (shared file or memory)
  2. extract and distribute snippets into buffers (optionally in parallel)

"""
from pathlib import Path

import numpy as np

from .job_tools import ChunkRecordingExecutor, ensure_n_jobs, _shared_job_kwargs_doc
from .core_tools import make_shared_array
from .job_tools import fix_job_kwargs



def extract_waveforms_to_buffers(recording, spikes, unit_ids, nbefore, nafter,
                                 mode='memmap', return_scaled=False, folder=None, dtype=None,
                                 sparsity_mask=None, copy=False, **job_kwargs):
    """
    Allocate buffers (memmap or or shared memory) and then distribute every waveform into theses buffers.

    Same as calling allocate_waveforms_buffers() and then distribute_waveforms_to_buffers().

    Important note: for the "shared_memory" mode wfs_arrays_info contains reference to
    the shared memmory buffer, this variable must be reference as long as arrays as used.
    And this variable is also returned.
    To avoid this a copy to non shared memmory can be perform at the end.

    Parameters
    ----------
    recording: recording
        The recording object
    spikes: 1d numpy array with several fields
        Spikes handled as a unique vector.
        This vector can be obtained with: `spikes = Sorting.to_spike_vector()`
    unit_ids: list ot numpy
        List of unit_ids
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    mode: str
        Mode to use ('memmap' | 'shared_memory')
    return_scaled: bool
        Scale traces before exporting to buffer or not.
    folder: str or path
        In case of memmap mode, folder to save npy files
    dtype: numpy.dtype
        dtype for waveforms buffer
    sparsity_mask: None or array of bool
        If not None shape must be must be (len(unit_ids), len(channel_ids))
    copy: bool
        If True (default), the output shared memory object is copied to a numpy standard array.
        If copy=False then wfs_arrays_info is also return. Please keep in mind that wfs_arrays_info 
        need to be referenced as long as wfs_arrays will be used otherwise it will be very hard to debug.
    {}
    
    Returns
    -------
    wfs_arrays: dict of arrays
        Arrays for all units (memmap or shared_memmep)

    wfs_arrays_info: dict of info
        Optionally return in case of shared_memory if copy=False.
        Dictionary to "construct" array in workers process (memmap file or sharemem info)
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    if dtype is None:
        if return_scaled:
            dtype = recording.get_dtype()
        else:
            dtype = 'float32'
    dtype = np.dtype(dtype)


    wfs_arrays, wfs_arrays_info = allocate_waveforms_buffers(recording, spikes, unit_ids, nbefore, nafter, mode=mode,
                                                             folder=folder, dtype=dtype, sparsity_mask=sparsity_mask)
    
    distribute_waveforms_to_buffers(recording, spikes, unit_ids, wfs_arrays_info, nbefore, nafter, return_scaled,
                                    mode=mode, sparsity_mask=sparsity_mask, **job_kwargs)

    if mode == 'memmap':
        return wfs_arrays
    elif mode == 'shared_memory':
        if copy:
            wfs_arrays = {unit_id: arr.copy() for unit_id, arr in wfs_arrays.items()}
            # clear shared mem buffer
            del wfs_arrays_info
            return wfs_arrays
        else:
            return wfs_arrays, wfs_arrays_info


extract_waveforms_to_buffers.__doc__ = extract_waveforms_to_buffers.__doc__.format(_shared_job_kwargs_doc)


def allocate_waveforms_buffers(recording, spikes, unit_ids, nbefore, nafter, mode='memmap', folder=None, dtype=None,
                               sparsity_mask=None):
    """
    Allocate memmap or shared memory buffers before snippet extraction.

    Important note: for the shared memory mode wfs_arrays_info contains reference to
    the shared memmory buffer, this variable must be reference as long as arrays as used.

    Parameters
    ----------
    recording: recording
        The recording object
    spikes: 1d numpy array with several fields
        Spikes handled as a unique vector.
        This vector can be obtained with: `spikes = Sorting.to_spike_vector()`
    unit_ids: list ot numpy
        List of unit_ids
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    mode: str
        Mode to use ('memmap' | 'shared_memory')
    folder: str or path
        In case of memmap mode, folder to save npy files
    dtype: numpy.dtype
        dtype for waveforms buffer
    sparsity_mask: None or array of bool
        If not None shape must be must be (len(unit_ids), len(channel_ids)

    Returns
    -------
    wfs_arrays: dict of arrays
        Arrays for all units (memmap or shared_memmep
    wfs_arrays_info: dict of info
        Dictionary to "construct" array in workers process (memmap file or sharemem)
    """

    nsamples = nbefore + nafter

    dtype = np.dtype(dtype)
    if mode == 'shared_memory':
        assert folder is None
    else:
        folder = Path(folder)

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

        if mode == 'memmap':
            filename = str(folder / f'waveforms_{unit_id}.npy')
            arr = np.lib.format.open_memmap(
                filename, mode='w+', dtype=dtype, shape=shape)
            wfs_arrays[unit_id] = arr
            wfs_arrays_info[unit_id] = filename
        elif mode == 'shared_memory':
            if n_spikes == 0:
                arr = np.zeros(shape, dtype=dtype)
                shm = None
                shm_name = None
            else:
                arr, shm = make_shared_array(shape, dtype)
                shm_name = shm.name
            wfs_arrays[unit_id] = arr
            wfs_arrays_info[unit_id] = (shm, shm_name, dtype.str, shape)
        else:
            raise ValueError('allocate_waveforms_buffers bad mode')

    return wfs_arrays, wfs_arrays_info


def distribute_waveforms_to_buffers(recording, spikes, unit_ids, wfs_arrays_info, nbefore, nafter, return_scaled,
                                    mode='memmap', sparsity_mask=None, **job_kwargs):
    """
    Distribute snippets of traces into corresponding buffers.

    Buffers must be pre-allocated with the `allocate_waveforms_buffers()` function.

    Important note, for "shared_memory" mode wfs_arrays_info contain reference to
    the shared memmory buffer, this variable must be reference as long as arrays as used.

    Parameters
    ----------
    recording: recording
        The recording object
    spikes: 1d numpy array with several field
        Spikes handled as a unique vector.
        This vector can be spikes = Sorting.to_spike_vector()
    unit_ids: list ot numpy
        List of unit_ids
    wfs_arrays_info: dict
        Dictionary to "construct" array in workers process (memmap file or sharemem)
    nbefore: int
        N samples before spike
    nafter: int
        N samples after spike
    return_scaled: bool
        Scale traces before exporting to buffer or not.
    mode: str
        Mode to use ('memmap' | 'shared_memory')
    sparsity_mask: None or array of bool
        If not None shape must be must be (len(unit_ids), len(channel_ids)

    {}

    """
    job_kwargs = fix_job_kwargs(job_kwargs)
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
    init_args = init_args + (unit_ids, spikes, wfs_arrays_info, nbefore,
                             nafter, return_scaled, inds_by_unit, mode, sparsity_mask)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args, job_name=f'extract waveforms {mode}',
                                       **job_kwargs)
    processor.run()


distribute_waveforms_to_buffers.__doc__ = distribute_waveforms_to_buffers.__doc__.format(_shared_job_kwargs_doc)


# used by ChunkRecordingExecutor
def _init_worker_waveform_extractor(recording, unit_ids, spikes, wfs_arrays_info, nbefore, nafter, return_scaled,
                                    inds_by_unit, mode, sparsity_mask):
    # create a local dict per worker
    worker_ctx = {}
    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)
    worker_ctx['recording'] = recording

    if mode == 'memmap':
        # in memmap mode we have the "too many open file" problem with linux
        # memmap file will be open on demand and not globally per worker
        worker_ctx['wfs_arrays_info'] = wfs_arrays_info
    elif mode == 'shared_memory':

        from multiprocessing.shared_memory import SharedMemory
        wfs_arrays = {}
        shms = {}
        for unit_id, (shm, shm_name, dtype, shape) in wfs_arrays_info.items():
            if shm_name is None:
                arr = np.zeros(shape=shape, dtype=dtype)
            else:
                shm = SharedMemory(shm_name)
                arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            wfs_arrays[unit_id] = arr
            # we need a reference to all sham otherwise we get segment fault!!!
            shms[unit_id] = shm
        worker_ctx['shms'] = shms
        worker_ctx['wfs_arrays'] = wfs_arrays

    worker_ctx['unit_ids'] = unit_ids
    worker_ctx['spikes'] = spikes
    
    worker_ctx['nbefore'] = nbefore
    worker_ctx['nafter'] = nafter
    worker_ctx['return_scaled'] = return_scaled
    worker_ctx['inds_by_unit'] = inds_by_unit
    worker_ctx['sparsity_mask'] = sparsity_mask
    worker_ctx['mode'] = mode
    

    return worker_ctx


# used by ChunkRecordingExecutor
def _waveform_extractor_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    recording = worker_ctx['recording']
    unit_ids = worker_ctx['unit_ids']
    spikes = worker_ctx['spikes']
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
        while (in_seg_spikes[i0]['sample_ind'] - nbefore) < 0 and (i0 != i1):
            i0 = i0 + 1
        while (in_seg_spikes[i1-1]['sample_ind'] + nafter) > seg_size and (i0 != i1):
            i1 = i1 - 1

    # slice in absolut in spikes vector
    l0 = i0 + s0
    l1 = i1 + s0

    if l1 > l0:
        start = spikes[l0]['sample_ind'] - nbefore
        end = spikes[l1-1]['sample_ind'] + nafter

        # load trace in memory
        traces = recording.get_traces(start_frame=start, end_frame=end, segment_index=segment_index,
                                      return_scaled=return_scaled)

        for unit_ind, unit_id in enumerate(unit_ids):
            # find pos
            inds = inds_by_unit[unit_id]
            in_chunk_pos,  = np.nonzero((inds >= l0) & (inds < l1))
            if in_chunk_pos.size ==0:
                continue
            
            if worker_ctx['mode'] == 'memmap':
                # open file in demand (and also autoclose it after)
                filename = worker_ctx['wfs_arrays_info'][unit_id]
                wfs = np.load(str(filename), mmap_mode='r+')
            elif worker_ctx['mode'] == 'shared_memory':
                wfs = worker_ctx['wfs_arrays'][unit_id]

            for pos in in_chunk_pos:
                sample_ind = spikes[inds[pos]]['sample_ind']
                wf = traces[sample_ind - start -
                            nbefore:sample_ind - start + nafter, :]

                if sparsity_mask is None:
                    wfs[pos, :, :] = wf
                else:
                    wfs[pos, :, :] = wf[:, sparsity_mask[unit_ind]]
