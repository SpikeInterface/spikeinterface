from pathlib import Path
import tempfile

import numpy as np
from spikeinterface.extractors import read_alf_sorting


def test_alf_sorting_extractor():
    """
    Here we generate spike train with 3 clusters in a very basic ALF format\
    and read it with the spikeinterface extractor
    """
    rec_len_secs, firing_rate, fs = (1000, 123, 30_000)
    spike_times = []
    spike_clusters = []
    for i, fr in enumerate([123, 78, 145]):
        st = np.cumsum(-np.log(np.random.rand(int(rec_len_secs * firing_rate * 1.5))) / firing_rate)
        st = st[: np.searchsorted(st, rec_len_secs)]
        spike_times.append(st)
        spike_clusters.append(st * 0 + i)
    spike_times = np.concatenate(spike_times)
    ordre = np.argsort(spike_times)
    spike_times = spike_times[ordre]
    spike_clusters = np.concatenate(spike_clusters)[ordre]

    with tempfile.TemporaryDirectory() as td:
        folder_path = Path(td)
        np.save(folder_path.joinpath("spikes.samples.npy"), (spike_times * fs).astype(int))
        np.save(folder_path.joinpath("spikes.clusters.npy"), spike_clusters)
        np.save(folder_path.joinpath("clusters.channels.npy"), np.arange(3))

        sorting = read_alf_sorting(folder_path, sampling_frequency=fs)
        assert sorting.get_unit_spike_train(0).size > 50_000
