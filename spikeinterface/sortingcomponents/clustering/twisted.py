# """Sorting components: clustering"""
from pathlib import Path

import numpy as np
try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

import random, string, os
from spikeinterface.core import get_global_tmp_folder
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler
from spikeinterface.toolkit import get_channel_distances
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers

class TwistedClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """
    _default_params = {
        "peak_locations" : None,
        "noise_levels" : None,
        "peak_localization_kwargs" : {"method" : "center_of_mass"},
        "hdbscan_kwargs": {"min_cluster_size" : 100,  "allow_single_cluster" : True, "core_dist_n_jobs" : -1},
        "debug" : False,
        "tmp_folder" : None,
        'radius_um' : 50,
        'ms_before': 1.5,
        'ms_after': 2.5,
        'waveform_mode' : 'memmap',
        'tmp_folder' : None,
        'job_kwargs' : {'n_jobs' : -1, 'chunk_memory' : '10M'},
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

        assert d['noise_levels'] is not None, "twisted clustering needs the noise levels"


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

        num_chans = recording.get_num_channels()
        sparsity_mask = np.zeros((peaks.size, num_chans), dtype='bool')
        chan_locs = recording.get_channel_locations()
        unit_inds = range(num_chans)
        chan_distances = get_channel_distances(recording)
        spikes['unit_ind'] = np.argmin(np.linalg.norm(chan_locs - locations[:, np.newaxis, :], axis=2), 1) 

        for main_chan in unit_inds:
            closest_chans, = np.nonzero(chan_distances[main_chan, :] <= params['radius_um'])
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

        wfs_arrays = extract_waveforms_to_buffers(recording, spikes, unit_inds, nbefore, nafter,
                                mode=params['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=recording.get_dtype(),
                                sparsity_mask=sparsity_mask,  copy=(params['waveform_mode'] == 'shared_memory'),
                                **params['job_kwargs'])

        energies = np.zeros((len(locations), 1))
        ptps = np.zeros((len(locations), 1))
        dist_ptps = np.zeros((len(locations), 1))

        #nb_ptps = 2
        #several_ptps = np.zeros((len(locations), nb_ptps))

        import scipy

        hanning_window = np.zeros(nbefore+nafter)
        hanning_window[:nbefore] = np.hanning(2*nbefore)[:nbefore]
        hanning_window[nbefore:] = np.hanning(2*nafter)[nafter:]

        hanning_window = hanning_window[:, np.newaxis]

        for main_chan, waveforms in wfs_arrays.items():
            idx = np.where(spikes['unit_ind'] == main_chan)[0]

            channels, = np.nonzero(sparsity_mask[main_chan])
            #idx_sorted = np.argsort(chan_distances[main_chan, channels])
            #closest_channels = idx_sorted[:nb_ptps]

            if len(waveforms) > 0:
                waveforms = waveforms*hanning_window
                all_ptps = np.ptp(waveforms, axis=1)
                ptps[idx, 0] = np.max(all_ptps, axis=1)

                ptp_channels = channels[np.argmax(np.ptp(waveforms, axis=1), axis=1)]
                #print(chan_locs[ptp_channels].shape, locations[idx].shape)
                #print(np.linalg.norm(chan_locs[ptp_channels] - locations[idx]))
                dist_ptps[idx, 0] = np.linalg.norm(chan_locs[ptp_channels] - locations[idx], axis=1)

                #ptps = np.ptp(waveforms, axis=1)
                #several_ptps[idx, :len(closest_channels)] = ptps[:, closest_channels]
                #data = waveforms.copy()
                #empty_channels = np.std(waveforms, axis=1) < 0.25*params['noise_levels'][channels]

                energies[idx, 0] = np.linalg.norm(waveforms, axis=(1,2))


        #preprocessing = MaxAbsScaler()
        #locations = preprocessing.fit_transform(locations)
        #preprocessing = QuantileTransformer(output_distribution='uniform')
        #several_ptps = preprocessing.fit_transform(several_ptps)
        #preprocessing = QuantileTransformer(output_distribution='uniform')
        #all_energies = preprocessing.fit_transform(all_energies)
        # weights = np.zeros((locations.shape[1] + nb_ptps + 1))
        # weights[:locations.shape[1]] = 1
        # weights[locations.shape[1]:] = 1/nb_ptps
        #to_cluster_from = np.hstack((locations, several_ptps, all_energies))
        #to_cluster_from *= weights

        to_cluster_from = np.hstack((locations, ptps, dist_ptps, energies))
        preprocessing = QuantileTransformer(output_distribution='uniform')
        to_cluster_from = preprocessing.fit_transform(to_cluster_from)

        clustering = hdbscan.hdbscan(to_cluster_from, **d['hdbscan_kwargs'])
        peak_labels = clustering[0]
        
        labels = np.unique(peak_labels)
        labels = labels[labels>=0]

        # print('Computing validation metrics...')
        # from .dbcv import DBCV
        # for i in labels:
        #     mask = peak_labels == i
        #     print(i, DBCV(to_cluster_from[mask], peak_labels[mask]))

        return labels, peak_labels
