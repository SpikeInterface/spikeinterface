import pytest
import numpy as np

from spikeinterface import download_dataset
from spikeinterface.sortingcomponents import detect_peaks, localize_peaks

from spikeinterface.extractors import MEArecRecordingExtractor


def test_localize_peaks():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(
        repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    peaks = detect_peaks(recording, method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=False, progress_bar=False)
    
    list_locations = []
    
    peak_locations = localize_peaks(recording, peaks, method='center_of_mass',
                                    chunk_size=10000, verbose=True, progress_bar=False)
    assert peaks.size == peak_locations.shape[0]
    #~ list_locations.append(('com', peak_locations))

    peak_locations = localize_peaks(recording, peaks, method='monopolar_triangulation',
                                    method_kwargs=dict(optimizer='least_square'),
                                    n_jobs=1, chunk_size=10000, verbose=True, progress_bar=True)
    assert peaks.size == peak_locations.shape[0]
    #~ list_locations.append(('least_square', peak_locations))

    peak_locations = localize_peaks(recording, peaks, method='monopolar_triangulation', 
                                    method_kwargs=dict(optimizer='minimize_with_log_penality'),
                                    n_jobs=1, chunk_size=10000, verbose=True, progress_bar=True)
    assert peaks.size == peak_locations.shape[0]
    #~ list_locations.append(('minimize_with_log_penality', peak_locations))


    # DEBUG
    #~ import MEArec
    #~ recgen = MEArec.load_recordings(recordings=local_path, return_h5_objects=True,
    #~ check_suffix=False,
    #~ load=['recordings', 'spiketrains', 'channel_positions'],
    #~ load_waveforms=False)
    #~ soma_positions = np.zeros((len(recgen.spiketrains), 3), dtype='float32')
    #~ for i, st in enumerate(recgen.spiketrains):
        #~ soma_positions[i, :] = st.annotations['soma_position']
    #~ import matplotlib.pyplot as plt
    #~ import spikeinterface.widgets as sw
    #~ from probeinterface.plotting import plot_probe
    #~ for title, peak_locations in list_locations:
        #~ probe = recording.get_probe()
        #~ fig, axs = plt.subplots(ncols=2, sharey=True)
        #~ ax = axs[0]
        #~ ax.set_title(title)
        #~ plot_probe(probe, ax=ax)
        #~ ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.5)
        #~ ax.set_xlabel('x')
        #~ ax.set_ylabel('y')
        #~ # MEArec is "yz" in 2D
        #~ ax.scatter(soma_positions[:, 1], soma_positions[:, 2], color='g', s=20, marker='*')
        #~ if 'z' in peak_locations.dtype.fields:
            #~ ax = axs[1]
            #~ ax.scatter(peak_locations['z'], peak_locations['y'], color='k', s=1, alpha=0.5)
            #~ ax.set_xlabel('z')
    #~ plt.show()



if __name__ == '__main__':
    test_localize_peaks()
