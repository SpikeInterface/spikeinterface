# """Sorting components: clustering"""
from pathlib import Path

import numpy as np
try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

class PositionAndPCAClustering:
    """
    Perform a hdbscan clustering on peak position then apply locals
    PCA on waveform + hdbscan on every spatial clustering to check
    if there a need to oversplit.
    
    """
    _default_params = {
        'peak_locations' : None,
        'peak_localization_kwargs' : {'method' : 'center_of_mass'},
        'ms_before': 1.5,
        'ms_after': 2.5,
        'n_components_by_channel' : 3,
        'n_components': 5,
        'job_kwargs' : {'n_jobs' : 1, 'chunk_memory' : '10M', 'progress_bar' : False},
        'hdbscan_kwargs_spatial': {'min_cluster_size' : 20, 'allow_single_cluster' : True},
        'hdbscan_kwargs_pca': {'min_cluster_size' : 10, 'allow_single_cluster' : True},
        'waveform_mode': 'memmap',
        'local_radius_um' : 50.,
        'noise_size' : 300,
        'debug' : False,
        'tmp_folder' : None,
#        'auto_merge_num_shift': 3,
#        'auto_merge_quantile_limit': 0.8, 
#        'ratio_num_channel_intersect': 0.5,
    }

    @classmethod
    def _check_params(cls, recording, peaks, params):
        d = params
        params2 = params.copy()
        
        tmp_folder = params['tmp_folder']
        if d['waveform_mode'] == 'memmap':
            if tmp_folder is None:
                name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                tmp_folder = get_global_tmp_folder() / f'PositionAndPcaClustering_{name}'
            else:
                tmp_folder = Path(tmp_folder)
            tmp_folder.mkdir()
            params2['tmp_folder'] = tmp_folder
        elif d['waveform_mode'] ==  'shared_memory':
            assert tmp_folder is None, 'temp_folder must be None for shared_memory'
        else:
            raise ValueError('shared_memory')        
        
        return params2
    
    @classmethod
    def _initialize_folder(cls, recording, peaks, params):
        d = params
        tmp_folder = params['tmp_folder']
        
        num_chans = recording.channel_ids.size
        
        # important sparsity is 2 times radius sparsity because closest channel will be 1 time radius
        chan_distances = get_channel_distances(recording)
        sparsity_mask = np.zeros((num_chans, num_chans), dtype='bool')
        for c in range(num_chans):
            chans, = np.nonzero(chan_distances[c, :] <= d['radius_um'])
            sparsity_mask[c, chans] = True
        
        # create a new peak vector to extract waveforms
        dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
        peaks2 = np.zeros(peaks.size, dtype=dtype)
        peaks2['sample_ind'] = peaks['sample_ind']
        peaks2['unit_ind'] = peaks['channel_ind']
        peaks2['segment_ind'] = peaks['segment_ind']
        
        fs = recording.get_sampling_frequency()
        dtype = recording.get_dtype()
        
        nbefore = int(d['ms_before'] * fs / 1000.)
        nafter = int(d['ms_after'] * fs / 1000.)
        
        if tmp_folder is None:
            wf_folder = None
        else:
            wf_folder = tmp_folder / 'waveforms_pre_cluster'
            wf_folder.mkdir()
            
        ids = np.arange(num_chans, dtype='int64')
        wfs_arrays = extract_waveforms_to_buffers(recording, peaks2, ids, nbefore, nafter,
                                mode=d['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=dtype,
                                sparsity_mask=sparsity_mask, copy=(d['waveform_mode'] == 'shared_memory'),
                                **d['job_kwargs'])

        # noise
        noise = get_random_data_chunks(recording, return_scaled=False,
                        num_chunks_per_segment=d['noise_size'], chunk_size=nbefore+nafter, concatenated=False, seed=None)
        noise = np.stack(noise, axis=0)


        return wfs_arrays, sparsity_mask, noise

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, 'sliding_hdbscan clustering need hdbscan to be installed'

        params = cls._check_params(recording, peaks, params)
        wfs_arrays, sparsity_mask, noise = cls._initialize_folder(recording, peaks, params)

        tmp_folder = d['tmp_folder']
        if tmp_folder is not None:
            tmp_folder = Path(tmp_folder)
            tmp_folder.mkdir(exist_ok=True)
        
        # step1 : clustering on peak location
        peak_locations = d['peak_locations']
        
        location_keys = ['x', 'y']
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)
                
        clustering = hdbscan.hdbscan(locations, **d['hdbscan_kwargs_spatial'])
        spatial_peak_labels = clustering[0]

        spatial_labels = np.unique(spatial_peak_labels)
        spatial_labels = spatial_labels[spatial_labels>=0]

        # if d['debug']:
        #     import matplotlib.pyplot as plt
        #     import spikeinterface.full as si
        #     fig1, ax = plt.subplots()
        #     kwargs = dict(probe_shape_kwargs=dict(facecolor='w', edgecolor='k', lw=0.5, alpha=0.3),
        #                             contacts_kargs = dict(alpha=0.5, edgecolor='k', lw=0.5, facecolor='w'))
        #     si.plot_probe_map(recording, ax=ax, **kwargs)
        #     ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, color='k')

        #     fig2, ax = plt.subplots()
        #     si.plot_probe_map(recording, ax=ax, **kwargs)
        #     ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, c=spatial_peak_labels)

        #     if tmp_folder is not None:
        #         fig1.savefig(tmp_folder / 'peak_locations.pdf')
        #         fig2.savefig(tmp_folder / 'peak_locations_clustered.pdf')


        # step2 : extract waveform by cluster
        spatial_keep, = np.nonzero(spatial_peak_labels >= 0)
        #~ num_keep = np.sum(spatial_keep)
        keep_peak_labels = spatial_peak_labels[spatial_keep]
        
        peak_dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
        peaks2 = np.zeros(spatial_keep.size, dtype=peak_dtype)
        peaks2['sample_ind'] = peaks['sample_ind'][spatial_keep]
        peaks2['segment_ind'] = peaks['segment_ind'][spatial_keep]

        num_chans = recording.get_num_channels()
        sparsity_mask = np.zeros((spatial_labels.size, num_chans), dtype='bool')
        chan_locs = recording.get_channel_locations()
        chan_distances = get_channel_distances(recording)
        for l, label in enumerate(spatial_labels):
            mask = keep_peak_labels == label
            peaks2['unit_ind'][mask] = l

            center = np.mean(locations[spatial_keep][mask], axis=0)
            main_chan = np.argmin(np.linalg.norm(chan_locs - center[np.newaxis, :], axis=1))
            
            # TODO take a radius that depend on the cluster dispertion itself
            closest_chans, = np.nonzero(chan_distances[main_chan, :] <= params['local_radius_um'])
            sparsity_mask[l, closest_chans] = True
        
        if d['waveform_mode'] == 'shared_memory':
            wf_folder = None
        else:
            assert tmp_folder is not None
            wf_folder = tmp_folder / 'waveforms_pre_cluster'
            wf_folder.mkdir()

        fs = recording.get_sampling_frequency()
        nbefore = int(d['ms_before'] * fs / 1000.)
        nafter = int(d['ms_after'] * fs / 1000.)

        ids = np.arange(num_chans, dtype='int64')
        wfs_arrays = extract_waveforms_to_buffers(recording, peaks2, spatial_labels, nbefore, nafter,
                                mode=d['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=recording.get_dtype(),
                                sparsity_mask=sparsity_mask,  copy=(d['waveform_mode'] == 'shared_memory'),
                                **d['job_kwargs'])

        noise = get_random_data_chunks(recording, return_scaled=False,
                        num_chunks_per_segment=d['noise_size'], chunk_size=nbefore+nafter, concatenated=False, seed=None)
        noise = np.stack(noise, axis=0)

        split_peak_labels, main_channels = auto_split_clustering(wfs_arrays, sparsity_mask, spatial_labels, keep_peak_labels, nbefore, nafter, noise, 
                n_components_by_channel=d['n_components_by_channel'],
                n_components=d['n_components'],
                hdbscan_params=d['hdbscan_params_pca'],
                probability_thr=d['probability_thr'],
                debug=d['debug'],
                debug_folder=tmp_folder,
                )
        
        peak_labels = -2 * np.ones(peaks.size, dtype=np.int64)
        peak_labels[spatial_keep] = split_peak_labels
        
        # auto clean
        pre_clean_labels = np.unique(peak_labels)
        pre_clean_labels = pre_clean_labels[pre_clean_labels>=0]
        #~ print('labels before auto clean', pre_clean_labels.size, pre_clean_labels)

        peaks3 = np.zeros(peaks.size, dtype=peak_dtype)
        peaks3['sample_ind'] = peaks['sample_ind']
        peaks3['segment_ind'] = peaks['segment_ind']
        peaks3['unit_ind'][:] = -1
        sparsity_mask3 = np.zeros((pre_clean_labels.size, num_chans), dtype='bool')
        for l, label in enumerate(pre_clean_labels):
            peaks3['unit_ind'][peak_labels == label] = l
            main_chan = main_channels[label]
            closest_chans, = np.nonzero(chan_distances[main_chan, :] <= params['local_radius_um'])
            sparsity_mask3[l, closest_chans] = True
        

        if d['waveform_mode'] == 'shared_memory':
            wf_folder = None
        else:
            if tmp_folder is not None:
                wf_folder = tmp_folder / 'waveforms_pre_autoclean'
                wf_folder.mkdir()
        
        wfs_arrays3 = extract_waveforms_to_buffers(recording, peaks3, pre_clean_labels, nbefore, nafter,
                                mode=d['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=recording.get_dtype(),
                                sparsity_mask=sparsity_mask3,  copy=(d['waveform_mode'] == 'shared_memory'),
                                **d['job_kwargs'])
        
        clean_peak_labels, peak_sample_shifts = auto_clean_clustering(wfs_arrays3, sparsity_mask3, pre_clean_labels, peak_labels, nbefore, nafter, chan_distances,
                                radius_um=d['local_radius_um'], auto_merge_num_shift=d['auto_merge_num_shift'],
                                auto_merge_quantile_limit=d['auto_merge_quantile_limit'], ratio_num_channel_intersect=d['ratio_num_channel_intersect'])
    
        
        # final
        labels = np.unique(clean_peak_labels)
        labels = labels[labels>=0]
        
        return labels, clean_peak_labels