# """Sorting components: clustering"""
from pathlib import Path

import shutil
import numpy as np
try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

import random, string, os
from spikeinterface.core import get_global_tmp_folder, get_noise_levels, get_channel_distances
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import remove_duplicates, remove_duplicates_via_matching
from spikeinterface.core import NumpySorting
from spikeinterface.core import extract_waveforms
from spikeinterface.sortingcomponents.features_from_peaks import compute_features_from_peaks, EnergyFeature


class RandomProjectionClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """
    _default_params = {
        "hdbscan_kwargs": {"min_cluster_size" : 100,  "allow_single_cluster" : True, "core_dist_n_jobs" : -1, "cluster_selection_method" : "leaf"},
        "cleaning_kwargs" : {"similar_threshold" : 0.99, "sparsify_threshold" : 0.99},
        "local_radius_um" : 100,
        "max_spikes_per_unit" : 200,
        "nb_projections" : 10,
        "ms_before" : 1.5,
        "ms_after": 2.5,
        "cleaning": "cosine",
        "job_kwargs" : {"n_jobs" : -1, "chunk_memory" : "10M", "verbose" : True, "progress_bar" : True},
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, 'twisted clustering need hdbscan to be installed'

        d = params

        peak_dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]

        fs = recording.get_sampling_frequency()
        nbefore = int(params['ms_before'] * fs / 1000.)
        nafter = int(params['ms_after'] * fs / 1000.)
        num_samples = nbefore + nafter
        num_chans = recording.get_num_channels()

        features_list = ['random_projections']
        projections = np.random.randn(num_chans, d['nb_projections'])
        features_params = {'random_projections': {'local_radius_um' : params['local_radius_um'], 
        'projections' : projections}}

        features_data = compute_features_from_peaks(recording, peaks, features_list, features_params, 
            ms_before=d['ms_before'], ms_after=d['ms_after'], **params['job_kwargs'])

        hdbscan_data = features_data[0]

        import sklearn
        clustering = hdbscan.hdbscan(hdbscan_data, **d['hdbscan_kwargs'])
        peak_labels = clustering[0]

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        best_spikes = {}
        nb_spikes = 0

        all_indices = np.arange(0, peak_labels.size)

        for unit_ind in labels:
            mask = peak_labels == unit_ind
            data = hdbscan_data[mask]
            centroid = np.median(data, axis=0)
            distances = sklearn.metrics.pairwise_distances(centroid[np.newaxis, :], data)[0]
            best_spikes[unit_ind] = all_indices[mask][np.argsort(distances)[:params["max_spikes_per_unit"]]]
            nb_spikes += best_spikes[unit_ind].size

        spikes = np.zeros(nb_spikes, dtype=peak_dtype)

        mask = np.zeros(0, dtype=np.int32)
        for unit_ind in labels:
            mask = np.concatenate((mask, best_spikes[unit_ind]))

        idx = np.argsort(mask)
        mask = mask[idx]

        spikes['sample_ind'] = peaks[mask]['sample_ind']
        spikes['segment_ind'] = peaks[mask]['segment_ind']
        spikes['unit_ind'] = peak_labels[mask]

        cleaning_method = params["cleaning"]

        print("We found %d raw clusters, starting to clean with %s..." %(len(labels), cleaning_method))

        if cleaning_method == "cosine":

            wfs_arrays = extract_waveforms_to_buffers(recording, spikes, labels, nbefore, nafter,
                         mode='shared_memory', return_scaled=False, folder=None, dtype=recording.get_dtype(),
                         sparsity_mask=None,  copy=True,
                         **params['job_kwargs'])

            noise_levels = get_noise_levels(recording, return_scaled=False)
            labels, peak_labels = remove_duplicates(wfs_arrays, noise_levels, peak_labels, num_samples, num_chans, **params['cleaning_kwargs'])

        elif cleaning_method == "matching":
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = Path(os.path.join(get_global_tmp_folder(), name))

            sorting = NumpySorting.from_times_labels(spikes['sample_ind'], spikes['unit_ind'], fs)
            we = extract_waveforms(recording, sorting, tmp_folder, overwrite=True, ms_before=params['ms_before'], 
                ms_after=params['ms_after'], **params['job_kwargs'])
            labels, peak_labels = remove_duplicates_via_matching(we, peak_labels, job_kwargs=params['job_kwargs'])
            shutil.rmtree(tmp_folder)

        print("We kept %d non-duplicated clusters..." %len(labels))

        return labels, peak_labels
