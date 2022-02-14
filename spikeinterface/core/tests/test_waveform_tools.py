from pathlib import Path
import shutil
import numpy as np

from spikeinterface.core.testing_tools import generate_recording, generate_sorting
#~ from spikeinterface import WaveformExtractor, extract_waveforms
from spikeinterface.core.waveform_tools import allocate_waveforms, distribute_waveform_to_buffers


def _clean_all():
    folders = ["wf_rec1", "wf_rec2", "wf_rec3", "wf_sort2", "wf_sort3",
               "test_waveform_extractor", "we_filt",
               "test_extract_waveforms_1job", "test_extract_waveforms_2job",
               "test_extract_waveforms_returnscaled",
               "test_extract_waveforms_sparsity"]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)


def setup_module():
    _clean_all()


def teardown_module():
    _clean_all()



def test_waveform_tools():


    durations = [30, 40]
    sampling_frequency = 30000.

    # 2 segments
    num_channels = 2
    recording = generate_recording(num_channels=num_channels, durations=durations, 
                                   sampling_frequency=sampling_frequency)
    recording.annotate(is_filtered=True)
    folder_rec = "wf_rec1"
    recording = recording.save(folder=folder_rec)
    num_units = 15
    sorting = generate_sorting(num_units=num_units, sampling_frequency=sampling_frequency, durations=durations)

    # test with dump !!!!
    recording = recording.save()
    sorting = sorting.save()

    wf_folder = Path('test_waveform_tools')
    if wf_folder.is_dir():
        shutil.rmtree(wf_folder)
    wf_folder.mkdir()

    #~ we = WaveformExtractor.create(recording, sorting, folder)
    
    nbefore = int(3. * sampling_frequency / 1000.)
    nafter = int(4. * sampling_frequency / 1000.)
    
    print('nbefore', nbefore, 'nafter', nafter)
    
    dtype = recording.get_dtype()
    return_scaled = False
    #~ dtype = 'float32'
    #~ return_scaled = True
    
    
    spikes = sorting.to_spike_vector()
    #~ print(spikes)
    
    unit_ids = sorting.unit_ids
    wfs_arrays = allocate_waveforms(recording, spikes, unit_ids, nbefore, nafter, mode='memmap', folder=wf_folder, dtype=dtype)
    #~ print(wfs_arrays)
    
    job_kwargs = {}
    job_kwargs = {'n_jobs': 1, 'chunk_size': 3000}
    job_kwargs = {'n_jobs': 2, 'chunk_size': 3000}
    distribute_waveform_to_buffers(recording, spikes, unit_ids, wfs_arrays, nbefore, nafter, return_scaled, **job_kwargs)


    for unit_ind, unit_id in enumerate(unit_ids):
        wf = wfs_arrays[unit_id]
        assert wf.shape[0] == np.sum(spikes['unit_ind'] == unit_ind)
    
    

    #~ we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)

    #~ we.run_extract_waveforms(n_jobs=1, chunk_size=30000)
    #~ we.run_extract_waveforms(n_jobs=4, chunk_size=30000, progress_bar=True)
    import matplotlib.pyplot as plt
    for unit_ind, unit_id in enumerate(unit_ids):
        wf = wfs_arrays[unit_id]
        wf_flat =wf.swapaxes(1, 2).reshape(wf.shape[0], -1)
        fig, ax = plt.subplots()
        ax.plot(wf_flat)
        plt.show()




if __name__ == '__main__':
    setup_module()
    test_waveform_tools()
    