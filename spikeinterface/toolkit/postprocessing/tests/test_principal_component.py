import unittest
import shutil
from pathlib import Path

import pytest



from spikeinterface import extract_waveforms, WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.toolkit.postprocessing import WaveformPrincipalComponent, compute_principal_components


def setup_module():
    for folder in ('toy_rec', 'toy_sorting', 'toy_waveforms'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)
    
    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder='toy_rec')
    sorting = sorting.save(folder='toy_sorting')
    
    we = extract_waveforms(recording, sorting, 'toy_waveforms',
        ms_before=3., ms_after=4., max_spikes_per_unit=500,
        n_jobs=1, chunk_size=30000)


def test_WaveformPrincipalComponent():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    unit_ids = we.sorting.unit_ids
    num_channels = we.recording.get_num_channels()
    pc = WaveformPrincipalComponent(we)
    
    for mode in ('by_channel_local', 'by_channel_global'):
        pc.set_params(n_components=5, mode=mode)
        print(pc)
        pc.run()
        for i, unit_id in enumerate(unit_ids):
            comp = pc.get_components(unit_id)
            #~ print(comp.shape)
            assert comp.shape[1:] == (5, 4)

        #~ import matplotlib.pyplot as plt
        #~ cmap = plt.get_cmap('jet', len(unit_ids))
        #~ fig, axs = plt.subplots(ncols=num_channels)
        #~ for i, unit_id in enumerate(unit_ids):
            #~ comp = pca.get_components(unit_id)
            #~ print(comp.shape)
            #~ for chan_ind in range(num_channels):
                #~ ax = axs[chan_ind]
                #~ ax.scatter(comp[:, 0, chan_ind], comp[:, 1, chan_ind], color=cmap(i))
        #~ plt.show()
    
    for mode in ('concatenated', ):
        pc.set_params(n_components=5, mode=mode)
        print(pc)
        pc.run()
        for i, unit_id in enumerate(unit_ids):
            comp = pc.get_components(unit_id)
            assert comp.shape[1] == 5
            #~ print(comp.shape)
    
    all_labels, all_components = pc.get_all_components()
    
        #~ import matplotlib.pyplot as plt
        #~ cmap = plt.get_cmap('jet', len(unit_ids))
        #~ fig, ax = plt.subplots()
        #~ for i, unit_id in enumerate(unit_ids):
            #~ comp = pca.get_components(unit_id)
            #~ print(comp.shape)
            #~ ax.scatter(comp[:, 0], comp[:, 1], color=cmap(i))
        #~ plt.show()

def test_compute_principal_components():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    pc = compute_principal_components(we, load_if_exists=True)
    print(pc)


if __name__ == '__main__':
    #~ setup_module()
    
    #~ test_WaveformPrincipalComponent()
    test_compute_principal_components()
