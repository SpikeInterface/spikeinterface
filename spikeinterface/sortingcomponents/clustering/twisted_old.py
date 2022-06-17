# """Sorting components: clustering"""
from pathlib import Path
import random
import string
import os

import numpy as np
try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

from spikeinterface.core import get_global_tmp_folder
from spikeinterface.toolkit import get_channel_distances, get_random_data_chunks
from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers
from .clustering_tools import auto_clean_clustering, auto_split_clustering

class TwistedClustering:
    """
    Perform a hdbscan clustering on peak position then apply locals
    PCA on waveform + hdbscan on every spatial clustering to check
    if there a need to oversplit. Should be fairly close to spyking-circus
    clustering
    
    """
    _default_params = {
        'ms_before': 1.5,
        'ms_after': 1.5,
        'n_components': 5,
        'job_kwargs' : {'n_jobs' : -1, 'chunk_memory' : '10M', 'verbose' : True, 'progress_bar' : True},
        'hdbscan_kwargs': {'min_cluster_size' : 20, 'allow_single_cluster' : True, "core_dist_n_jobs" : -1},
        'waveform_mode': 'memmap',
        'n_neighbors' : 8,
        'debug' : False,
        'tmp_folder' : None,
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

        #res = PositionClustering(recording, peaks, params)

        assert HAVE_HDBSCAN, 'position_and_pca clustering need hdbscan to be installed'

        params = cls._check_params(recording, peaks, params)
        #wfs_arrays, sparsity_mask, noise = cls._initialize_folder(recording, peaks, params)

        peak_dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
        spikes = np.zeros(peaks.size, dtype=peak_dtype)
        spikes['sample_ind'] = peaks['sample_ind']
        spikes['segment_ind'] = peaks['segment_ind']
        spikes['unit_ind'] = peaks['channel_ind']

        num_chans = recording.get_num_channels()
        unit_ids = np.arange(num_chans)
        sparsity_mask = np.zeros((num_chans, num_chans), dtype='bool')
        chan_locs = recording.get_channel_locations()
        chan_distances = get_channel_distances(recording)
        channels_neighbors = np.argsort(chan_distances, axis=1)[:, :params['n_neighbors']]
        for main_chan in unit_ids:
            sparsity_mask[main_chan, channels_neighbors[main_chan]] = True 


        if params['waveform_mode'] == 'shared_memory':
            wf_folder = None
        else:
            assert params['tmp_folder'] is not None
            wf_folder = params['tmp_folder'] / 'sparse_snippets'
            wf_folder.mkdir()

        fs = recording.get_sampling_frequency()
        nbefore = int(params['ms_before'] * fs / 1000.)
        nafter = int(params['ms_after'] * fs / 1000.)

        ids = np.arange(num_chans, dtype='int64')
        wfs_arrays = extract_waveforms_to_buffers(recording, spikes, unit_ids, nbefore, nafter,
                                mode=params['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=recording.get_dtype(),
                                sparsity_mask=sparsity_mask,  copy=(params['waveform_mode'] == 'shared_memory'),
                                **params['job_kwargs'])

        result = {}
        import sklearn.decomposition
        pca = sklearn.decomposition.IncrementalPCA(n_components=params['n_components'], whiten=True)
        hanning_winwdow = np.hanning(nbefore+nafter)[:, np.newaxis]

        for key, value in wfs_arrays.items():
            
            data = (value*hanning_winwdow)[:,::2,:]
            data = data.reshape(len(data), -1)
            
            pca.partial_fit(data)

        features = np.zeros((0, params['n_components']), dtype=np.float32)
        for key, value in wfs_arrays.items():
            data = (value*hanning_winwdow)[:,::2,:]
            data = data.reshape(len(data), -1)
            features = np.vstack((features, pca.transform(data)))

        locations = chan_locs[peaks['channel_ind']]
        to_cluster_from = np.hstack((locations, features))
        
        clustering = hdbscan.hdbscan(to_cluster_from, **params['hdbscan_kwargs'])
        peak_labels = clustering[0]
        
        labels = np.unique(peak_labels)
        labels = peak_labels[peak_labels >= 0]


        return labels, peak_labels