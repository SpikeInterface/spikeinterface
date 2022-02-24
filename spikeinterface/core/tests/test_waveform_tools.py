from pathlib import Path
import shutil
import platform

import numpy as np

from spikeinterface.core.testing_tools import generate_recording, generate_sorting
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers #allocate_waveforms_buffers, distribute_waveforms_to_buffers


def _clean_all():
    folders = ["wf_rec1", "test_waveform_tools", "test_waveform_tools_sparse",]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)


def setup_module():
    _clean_all()


def teardown_module():
    _clean_all()


def _check_all_wf_equal(list_wfs_arrays):
    wfs_arrays0 = list_wfs_arrays[0]
    for i, wfs_arrays in enumerate(list_wfs_arrays):
        for unit_id in wfs_arrays.keys():
            assert np.array_equal(wfs_arrays[unit_id], wfs_arrays0[unit_id])


def test_waveform_tools():


    durations = [30, 40]
    sampling_frequency = 30000.

    # 2 segments
    num_channels = 2
    recording = generate_recording(num_channels=num_channels, durations=durations, 
                                   sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    folder_rec = "wf_rec1"
    #~ recording = recording.save(folder=folder_rec)
    num_units = 15
    sorting = generate_sorting(num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)

    # test with dump !!!!
    recording = recording.save()
    sorting = sorting.save()

    nbefore = int(3. * sampling_frequency / 1000.)
    nafter = int(4. * sampling_frequency / 1000.)
    
    dtype = recording.get_dtype()
    # return_scaled = False
    
    spikes = sorting.to_spike_vector()
    
    unit_ids = sorting.unit_ids

    some_job_kwargs = [
        {},
        {'n_jobs': 1, 'chunk_size': 3000, 'progress_bar':True},
        {'n_jobs': 2, 'chunk_size': 3000, 'progress_bar':True},
    ]
    
    # memmap mode 
    list_wfs = []
    for j, job_kwargs in enumerate(some_job_kwargs):
        wf_folder = Path(f'test_waveform_tools_{j}')
        if wf_folder.is_dir():
            shutil.rmtree(wf_folder)
        wf_folder.mkdir()
        # wfs_arrays, wfs_arrays_info = allocate_waveforms_buffers(recording, spikes, unit_ids, nbefore, nafter, mode='memmap', folder=wf_folder, dtype=dtype)
        # distribute_waveforms_to_buffers(recording, spikes, unit_ids, wfs_arrays_info, nbefore, nafter, return_scaled, **job_kwargs)
        wfs_arrays = extract_waveforms_to_buffers(recording, spikes, unit_ids, nbefore, nafter,
                                mode='memmap', return_scaled=False, folder=wf_folder, dtype=dtype,
                                sparsity_mask=None,  copy=False, **job_kwargs)
        for unit_ind, unit_id in enumerate(unit_ids):
            wf = wfs_arrays[unit_id]
            assert wf.shape[0] == np.sum(spikes['unit_ind'] == unit_ind)
        list_wfs.append({unit_id: wfs_arrays[unit_id].copy() for unit_id in unit_ids})
    _check_all_wf_equal(list_wfs)
    

    # memory
    if platform.system() != 'Windows':
        # shared memory on windows is buggy...
        list_wfs = []
        for job_kwargs in some_job_kwargs:
            # wfs_arrays, wfs_arrays_info = allocate_waveforms_buffers(recording, spikes, unit_ids, nbefore, nafter, mode='shared_memory', folder=None, dtype=dtype)
            # distribute_waveforms_to_buffers(recording, spikes, unit_ids, wfs_arrays_info, nbefore, nafter, return_scaled, mode='shared_memory', **job_kwargs)
            wfs_arrays = extract_waveforms_to_buffers(recording, spikes, unit_ids, nbefore, nafter,
                                mode='shared_memory', return_scaled=False, folder=None, dtype=dtype,
                                sparsity_mask=None,  copy=True, **job_kwargs)
            for unit_ind, unit_id in enumerate(unit_ids):
                wf = wfs_arrays[unit_id]
                assert wf.shape[0] == np.sum(spikes['unit_ind'] == unit_ind)
            list_wfs.append({unit_id: wfs_arrays[unit_id].copy() for unit_id in unit_ids})
            # to avoid warning we need to first destroy arrays then sharedmemm object
            # del wfs_arrays
            # del wfs_arrays_info
        _check_all_wf_equal(list_wfs)

    
    # with sparsity
    wf_folder = Path('test_waveform_tools_sparse')
    if wf_folder.is_dir():
        shutil.rmtree(wf_folder)
    wf_folder.mkdir()
    
    
    sparsity_mask = np.random.randint(0, 2, size=(unit_ids.size, recording.channel_ids.size), dtype='bool')
    job_kwargs = {'n_jobs': 1, 'chunk_size': 3000, 'progress_bar':True}

    # wfs_arrays, wfs_arrays_info = allocate_waveforms_buffers(recording, spikes, unit_ids, nbefore, nafter, mode='memmap', folder=wf_folder, dtype=dtype, sparsity_mask=sparsity_mask)
    # distribute_waveforms_to_buffers(recording, spikes, unit_ids, wfs_arrays_info, nbefore, nafter, return_scaled, sparsity_mask=sparsity_mask, **job_kwargs)

    wfs_arrays = extract_waveforms_to_buffers(recording, spikes, unit_ids, nbefore, nafter,
                        mode='memmap', return_scaled=False, folder=wf_folder, dtype=dtype,
                        sparsity_mask=sparsity_mask,  copy=False, **job_kwargs)



if __name__ == '__main__':
    setup_module()
    test_waveform_tools()
    