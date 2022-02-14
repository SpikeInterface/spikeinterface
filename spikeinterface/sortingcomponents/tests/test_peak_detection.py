import pytest
import numpy as np

from spikeinterface import download_dataset, BaseSorting
from spikeinterface.sortingcomponents import detect_peaks

from spikeinterface.extractors import MEArecRecordingExtractor


def test_detect_peaks():

    repo = 'https://gin.g-node.org/NeuralEnsemble/ephy_testing_data'
    remote_path = 'mearec/mearec_test_10s.h5'
    local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=None)
    recording = MEArecRecordingExtractor(local_path)

    # by_channel
    peaks = detect_peaks(recording, method='by_channel',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=False)
    
    # by_channel
    sorting = detect_peaks(recording, method='by_channel',
                           peak_sign='neg', detect_threshold=5, n_shifts=2,
                           chunk_size=10000, verbose=1, progress_bar=False,
                           outputs="sorting")
    assert isinstance(sorting, BaseSorting)

    # locally_exclusive
    peaks = detect_peaks(recording, method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=False)

    # locally_exclusive + localization
    peaks = detect_peaks(recording, method='locally_exclusive',
                         peak_sign='neg', detect_threshold=5, n_shifts=2,
                         chunk_size=10000, verbose=1, progress_bar=True,
                         localization_dict=dict(method='center_of_mass', local_radius_um=150, ms_before=0.1, ms_after=0.3),
                         #localization_dict=dict(method='monopolar_triangulation', local_radius_um=150,
                         #                       ms_before=0.1, ms_after=0.3, max_distance_um=1000)
                         )
    assert 'x' in peaks.dtype.fields


    # DEBUG
    #~ sample_inds, chan_inds, amplitudes = peaks['sample_ind'], peaks['channel_ind'], peaks['amplitude']
    #~ import matplotlib.pyplot as plt
    #~ import spikeinterface.widgets as sw
    #~ chan_offset = 500
    #~ traces = recording.get_traces()
    #~ traces += np.arange(traces.shape[1])[None, :] * chan_offset
    #~ fig, ax = plt.subplots()
    #~ ax.plot(traces, color='k')
    #~ ax.scatter(sample_inds, chan_inds * chan_offset + amplitudes, color='r')
    #~ plt.show()

    #~ import matplotlib.pyplot as plt
    #~ import spikeinterface.widgets as sw
    #~ from probeinterface.plotting import plot_probe
    #~ probe = recording.get_probe()
    #~ fig, ax = plt.subplots()
    #~ plot_probe(probe, ax=ax)
    #~ ax.scatter(peaks['x'], peaks['z'], color='k', s=1, alpha=0.5)
    #~ # MEArec is "yz" in 2D
    #~ import MEArec
    #~ recgen = MEArec.load_recordings(recordings=local_path, return_h5_objects=True,
    #~ check_suffix=False,
    #~ load=['recordings', 'spiketrains', 'channel_positions'],
    #~ load_waveforms=False)
    #~ soma_positions = np.zeros((len(recgen.spiketrains), 3), dtype='float32')
    #~ for i, st in enumerate(recgen.spiketrains):
        #~ soma_positions[i, :] = st.annotations['soma_position']
    #~ ax.scatter(soma_positions[:, 1], soma_positions[:, 2], color='g', s=20, marker='*')
    #~ plt.show()


if __name__ == '__main__':
    test_detect_peaks()
