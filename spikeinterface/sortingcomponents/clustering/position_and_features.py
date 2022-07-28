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


class PositionAndFeaturesClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """
    _default_params = {
        "peak_locations" : None,
        "peak_localization_kwargs" : {"method" : "center_of_mass"},
        "hdbscan_kwargs": {"min_cluster_size" : 100,  "allow_single_cluster" : True, "core_dist_n_jobs" : -1, "cluster_selection_method" : "leaf"},
        "cleaning_kwargs" : {"similar_threshold" : 0.99, "sparsify_threshold" : 0.99},
        "tmp_folder" : None,
        "local_radius_um" : 50,
        "max_spikes_per_unit" : 100,
        "ms_before" : 1.5,
        "ms_after": 2.5,
        "cleaning": "cosine",
        "waveform_mode" : "memmap",
        "job_kwargs" : {"n_jobs" : -1, "chunk_memory" : "10M"},
    }

    @classmethod
    def _check_params(cls, recording, peaks, params):
        d = params
        params2 = params.copy()
        
        tmp_folder = params['tmp_folder']
        if params['waveform_mode'] == 'memmap':
            if tmp_folder is None:
                name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                tmp_folder = Path(os.path.join(get_global_tmp_folder(), name))
            else:
                tmp_folder = Path(tmp_folder)
            tmp_folder.mkdir()
            params2['tmp_folder'] = tmp_folder
        elif params['waveform_mode'] ==  'shared_memory':
            assert tmp_folder is None, 'tmp_folder must be None for shared_memory'
        else:
            raise ValueError('shared_memory')        
        
        return params2

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, 'twisted clustering need hdbscan to be installed'

        params = cls._check_params(recording, peaks, params)
        d = params

        if d['peak_locations'] is None:
            from spikeinterface.sortingcomponents.peak_localization import localize_peaks
            peak_locations = localize_peaks(recording, peaks, **d['peak_localization_kwargs'], **d['job_kwargs'])
        else:
            peak_locations = d['peak_locations']

        tmp_folder = d['tmp_folder']
        if tmp_folder is not None:
            tmp_folder.mkdir(exist_ok=True)
    
        location_keys = ['x', 'y']
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)

        peak_dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
        spikes = np.zeros(peaks.size, dtype=peak_dtype)
        spikes['sample_ind'] = peaks['sample_ind']
        spikes['segment_ind'] = peaks['segment_ind']
        spikes['unit_ind'] = peaks['channel_ind']

        num_chans = recording.get_num_channels()
        sparsity_mask = np.zeros((peaks.size, num_chans), dtype='bool')
        chan_locs = recording.get_channel_locations()
        unit_inds = range(num_chans)
        chan_distances = get_channel_distances(recording)
        
        for main_chan in unit_inds:
            closest_chans, = np.nonzero(chan_distances[main_chan, :] <= params['local_radius_um'])
            sparsity_mask[main_chan, closest_chans] = True

        if params['waveform_mode'] == 'shared_memory':
            wf_folder = None
        else:
            assert params['tmp_folder'] is not None
            wf_folder = params['tmp_folder'] / 'sparse_snippets'
            wf_folder.mkdir()

        fs = recording.get_sampling_frequency()
        nbefore = int(params['ms_before'] * fs / 1000.)
        nafter = int(params['ms_after'] * fs / 1000.)
        num_samples = nbefore + nafter

        wfs_arrays = extract_waveforms_to_buffers(recording, spikes, unit_inds, nbefore, nafter,
                                mode=params['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=recording.get_dtype(),
                                sparsity_mask=sparsity_mask,  copy=(params['waveform_mode'] == 'shared_memory'),
                                **params['job_kwargs'])


        n_loc = len(location_keys)
        hdbscan_data = np.zeros((len(locations), n_loc + 3), dtype=np.float32)
        hdbscan_data[:, :n_loc] = locations

        import scipy, sklearn

        #hanning_window = np.zeros(nbefore+nafter)
        #hanning_window[:nbefore] = np.hanning(2*nbefore)[:nbefore]
        #hanning_window[nbefore:] = np.hanning(2*nafter)[nafter:]

        #step_window = hanning_window[:, np.newaxis]
        #step_window = step_window[:, np.newaxis]

        for main_chan, waveforms in wfs_arrays.items():
            idx = np.where(spikes['unit_ind'] == main_chan)[0]

            channels, = np.nonzero(sparsity_mask[main_chan])
            #idx_sorted = np.argsort(chan_distances[main_chan, channels])
            #closest_channels = idx_sorted[:nb_ptps]

            if len(waveforms) > 0:
                #waveforms_filtered = scipy.signal.savgol_filter(waveforms, 11, 3 , axis=1)
                #waveforms = waveforms_filtered*(1 - step_window) + step_window*waveforms
                all_ptps = np.ptp(waveforms, axis=1)
                ptp_channels = channels[np.argmax(np.ptp(waveforms, axis=1), axis=1)]

                hdbscan_data[idx, n_loc] = np.max(all_ptps, axis=1)
                hdbscan_data[idx, n_loc + 1] = np.linalg.norm(chan_locs[ptp_channels] - locations[idx], axis=1)
                hdbscan_data[idx, n_loc + 2] = np.linalg.norm(waveforms, axis=(1,2))/np.sqrt(len(channels))


        preprocessing = QuantileTransformer(output_distribution='uniform')
        hdbscan_data = preprocessing.fit_transform(hdbscan_data)
        
        clustering = hdbscan.hdbscan(hdbscan_data, **d['hdbscan_kwargs'])
        peak_labels = clustering[0]

        labels = np.unique(peak_labels)
        labels = labels[labels>=0]

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

        if params['waveform_mode'] == 'shared_memory':
            wf_folder = None
        else:
            assert params['tmp_folder'] is not None
            wf_folder = params['tmp_folder'] / 'dense_snippets'
            wf_folder.mkdir()


        cleaning_method = params["cleaning"]

        print("We found %d raw clusters, starting to clean with %s..." %(len(labels), cleaning_method))

        if cleaning_method == "cosine":

            wfs_arrays = extract_waveforms_to_buffers(recording, spikes, labels, nbefore, nafter,
                         mode=params['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=recording.get_dtype(),
                         sparsity_mask=None,  copy=(params['waveform_mode'] == 'shared_memory'),
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
