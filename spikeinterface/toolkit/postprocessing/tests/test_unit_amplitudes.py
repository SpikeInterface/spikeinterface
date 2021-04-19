import pytest
import numpy as np

from spikeinterface import download_dataset, extract_waveforms
import spikeinterface.extractors as se
from spikeinterface.toolkit import get_unit_amplitudes
    
    
def test_get_unit_amplitudes():
    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = se.MEArecRecordingExtractor(local_path)
    sorting = se.MEArecSortingExtractor(local_path)
    
    
    we = extract_waveforms(recording, sorting, 'toy_waveforms',
        ms_before=1., ms_after=2., max_spikes_per_unit=500,
        n_jobs=1, chunk_size=30000, load_if_exists=True)
    

    amplitudes, segments = get_unit_amplitudes(we)
    print(amplitudes)
    print(segments)
    
    
    
    
    
    # DEBUG
    #~ import matplotlib.pyplot as plt
    #~ import spikeinterface.widgets as sw
    #~ chan_offset = 500
    #~ traces = recording.get_traces()
    #~ traces += np.arange(traces.shape[1])[None, :] * chan_offset
    #~ print(traces.shape)
    #~ fig, ax = plt.subplots()
    #~ ax.plot(traces, color='k')
    #~ ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, color='r')
    #~ plt.show()
    



if __name__ == '__main__':
    test_get_unit_amplitudes()
