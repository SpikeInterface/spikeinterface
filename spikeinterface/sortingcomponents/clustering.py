"""Sorting components: clustering"""
import random
import string
from pathlib import Path

import time

import numpy as np

import sklearn.decomposition

try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False


from ..core import get_global_tmp_folder
from ..toolkit import get_channel_distances, get_random_data_chunks
from ..core.waveform_tools import extract_waveforms_to_buffers


from .clustering_tools import auto_clean_clustering


def find_cluster_from_peaks(recording, peaks, method='stupid', method_kwargs={}, extra_outputs=False, **job_kwargs):
    """
    Find cluster from peaks.
    

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    peaks: WaveformExtractor
        The waveform extractor
    method: str
        Which method to use ('stupid' | 'XXXX')
    method_kwargs: dict, optional
        Keyword arguments for the chosen method
    extra_outputs: bool
        If True then debug is also return

    Returns
    -------
    labels: ndarray of int
        possible clusters list
    peak_labels: array of int
        peak_labels.shape[0] == peaks.shape[0]
    """
    
    assert method in clustering_methods, f'Method for clustering do not exists, should be in {list(clustering_methods.keys())}'
    
    method_class = clustering_methods[method]
    params = method_class._default_params.copy()
    params.update(**method_kwargs)
    
    labels, peak_labels = method_class.main_function(recording, peaks, params)
    
    if extra_outputs:
        raise NotImplementedError


    return labels, peak_labels
    


class StupidClustering:
    """
    Stupid clustering.
    peak are clustered from there channel detection
    So peak['channel_ind'] will be the peak_labels
    """
    _default_params = {
    }
    
    @classmethod
    def main_function(cls, recording, peaks, params):
        labels = np.arange(recording.get_num_channels(), dtype='int64')
        peak_labels = peaks['channel_ind']
        return labels, peak_labels


class PositionClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """
    _default_params = {
        "peak_locations" : None,
        "hdbscan_params": {"min_cluster_size" : 20,  "allow_single_cluster" : True, 'metric' : 'l2'},
        "probability_thr" : 0,
        "apply_norm" : True,
        "debug" : False,
        "tmp_folder" : None,
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        d = params

        assert d['peak_locations'] is not None, "peak_locations should not be None!"

        tmp_folder = d['tmp_folder']
        if tmp_folder is not None:
            tmp_folder.mkdir(exist_ok=True)

        peak_locations = d['peak_locations']
        
        location_keys = ['x', 'y']
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)
        
        if d['apply_norm']:
            locations_n = locations.copy()
            locations_n -= np.mean(locations_n, axis=0)
            locations_n /= np.std(locations_n, axis=0)

        
        clustering = hdbscan.hdbscan(locations_n, **d['hdbscan_params'])
        peak_labels = clustering[0]
        peak_labels_persistent = peak_labels.copy()
        cluster_probability = clustering[2]

        persistent_clusters, = np.nonzero(clustering[2] > d['probability_thr'])
        mask = np.in1d(peak_labels, persistent_clusters)
        peak_labels_persistent[~mask] = -2
        
        final_peak_labels = peak_labels_persistent
        labels = np.unique(peak_labels)
        labels =labels[labels>=0]

        if d['debug']:
            import matplotlib.pyplot as plt
            import spikeinterface.full as si
            fig1, ax = plt.subplots()
            kwargs = dict(probe_shape_kwargs=dict(facecolor='w', edgecolor='k', lw=0.5, alpha=0.3),
                                    contacts_kargs = dict(alpha=0.5, edgecolor='k', lw=0.5, facecolor='w'))
            si.plot_probe_map(recording, ax=ax, **kwargs)
            ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, color='k')

            fig1, ax = plt.subplots()
            si.plot_probe_map(recording, ax=ax, **kwargs)
            ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, c=peak_labels)

            if tmp_folder is not None:
                fig1.savefig(tmp_folder / 'peak_locations.png')
                fig2.savefig(tmp_folder / 'peak_locations_clustered.png')

        return np.unique(peak_labels), final_peak_labels


class SlidingHdbscanClustering:
    """
    This is a port of the tridesclous clustering.
    
    This internally make many local hdbscan clustering on 
    a local radius. The dimention reduction (features) is done on the fly.
    This is done iteractively.
    
    One advantage is that the high amplitude units do bias the PC after
    have been selected.
    
    This method is a bit slow
    """
    _default_params = {
        'waveform_mode': 'memmap',
        'tmp_folder': None,
        'ms_before': 1.5,
        'ms_after': 2.5,
        'noise_size' : 300,
        'min_spike_on_channel' : 5,
        'stop_explore_percent' : 0.05,
        'min_cluster_size' : 10,
        'radius_um': 50.,
        'n_components_by_channel': 4,
        'auto_merge_num_shift': 3,
        'auto_merge_quantile_limit': 0.8, 
        #~ 'auto_trash_misalignment_shift' : 4,
        
        'job_kwargs' : {}
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, 'SlidingHdbscanClustering need hdbscan to be installed'
        params = cls._check_params(recording, peaks, params)
        wfs_arrays, wfs_arrays_info, sparsity_mask, noise = cls._initlialize_folder(recording, peaks, params)
        peak_labels = cls._find_clusters(recording, peaks, wfs_arrays, sparsity_mask, noise, params)
        
        wfs_arrays2, sparsity_mask2 = cls._prepare_clean(recording, peaks, wfs_arrays, sparsity_mask, peak_labels, params)
        
        clean_peak_labels, peak_sample_shifts = cls._clean_cluster(recording, peaks, wfs_arrays2, sparsity_mask2, peak_labels, params)
        
        labels = np.unique(clean_peak_labels)
        labels = labels[labels >= 0]

        return labels, peak_labels

    @classmethod
    def _check_params(cls, recording, peaks, params):
        d = params
        params2 = params.copy()
        
        tmp_folder = params['tmp_folder']
        if d['waveform_mode'] ==  'memmap':
            if tmp_folder is None:
                name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                tmp_folder = get_global_tmp_folder() / f'SlidingHdbscanClustering_{name}'
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
    def _initlialize_folder(cls, recording, peaks, params):
        d = params
        tmp_folder = params['tmp_folder']
        
        num_chans = recording.channel_ids.size
        
        # important sparsity is 2 times radius sparsity because closest channel will be 1 time radius
        chan_distances = get_channel_distances(recording)
        sparsity_mask = np.zeros((num_chans, num_chans), dtype='bool')
        for c in range(num_chans):
            chans, = np.nonzero(chan_distances[c, :] <= ( 2 * d['radius_um']))
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
                                sparsity_mask=sparsity_mask,  copy=(d['waveform_mode'] == 'shared_memory'),
                                **d['job_kwargs'])

        # noise
        noise = get_random_data_chunks(recording, return_scaled=False,
                        num_chunks_per_segment=d['noise_size'], chunk_size=nbefore+nafter, concatenated=False, seed=None)
        noise = np.stack(noise, axis=0)


        return wfs_arrays, sparsity_mask, noise
    
    @classmethod
    def _find_clusters(cls, recording, peaks, wfs_arrays, sparsity_mask, noise, d):
        
        num_chans = recording.get_num_channels()
        fs = recording.get_sampling_frequency()
        nbefore = int(d['ms_before'] * fs / 1000.)
        nafter = int(d['ms_after'] * fs / 1000.)
        
        
        
        possible_channel_inds = np.unique(peaks['channel_ind'])
        
        # channel neighborhood
        chan_distances = get_channel_distances(recording)
        closest_channels = []
        for c in range(num_chans):
            chans, = np.nonzero(chan_distances[c, :] <= d['radius_um'])
            chans = np.intersect1d(possible_channel_inds, chans)
            closest_channels.append(chans)
        
        peak_labels = np.zeros(peaks.size, dtype='int64')
        
        # create amplitudes percentile vector
        # this help to explore channel starting with high amplitudes
        chan_amps = np.zeros(num_chans, dtype='float64')
        remain_count = np.zeros(num_chans, dtype='int64')
        remain_percent = np.zeros(num_chans, dtype='float64')
        total_count = np.zeros(num_chans, dtype='int64')
        for chan_ind in range(num_chans):
            total_count[chan_ind] = np.sum(peaks['channel_ind'] == chan_ind)

        # this force compute compute at forst loop
        prev_local_chan_inds = np.arange(num_chans, dtype='int64')
        
        actual_label = 1
        
        while True:
            print()
            print('actual_label', actual_label, 'remain', np.sum(peak_labels==0))

            # update ampltiude percentile and count peak by channel
            for chan_ind in prev_local_chan_inds:
                if total_count[chan_ind] == 0:
                    continue
                #~ inds, = np.nonzero(np.in1d(peaks['channel_ind'], closest_channels[chan_ind]) & (peak_labels==0))
                inds, = np.nonzero((peaks['channel_ind'] == chan_ind) & (peak_labels==0))
                if inds.size <= d['min_spike_on_channel']:
                    chan_amps[chan_ind] = 0.
                else:
                    amps = np.abs(peaks['amplitude'][inds])
                    chan_amps[chan_ind] = np.percentile(amps, 90)
                remain_count[chan_ind] = inds.size
                remain_percent[chan_ind] = remain_count[chan_ind] / total_count[chan_ind]
                if remain_percent[chan_ind] < d['stop_explore_percent']:
                    chan_amps[chan_ind] = 0.
            
            
            # get best channel
            if np.all(chan_amps == 0):
                break
            
            # try fist unexplore and high amplitude
            # local_chan_ind = np.argmax(chan_amps)
            local_chan_ind = np.argmax(chan_amps * remain_percent)
            local_chan_inds = closest_channels[local_chan_ind]
            
            # take waveforms not label yet for channel in radius
            t0 = time.perf_counter()
            wfs = []
            local_peak_ind = []
            for chan_ind in local_chan_inds:
                sel,  = np.nonzero(peaks['channel_ind'] == chan_ind)
                inds,  = np.nonzero(peak_labels[sel] == 0)
                local_peak_ind.append(sel[inds])
                # here a unit is a channel index!!!
                wfs_chan = wfs_arrays[chan_ind]
                
                # TODO: only for debug, remove later
                assert wfs_chan.shape[0] == sel.size
                
                wf_chans, = np.nonzero(sparsity_mask[chan_ind])
                # TODO: only for debug, remove later
                assert np.all(np.in1d(local_chan_inds, wf_chans))
                
                # none label spikes
                wfs_chan = wfs_chan[inds, :, :]
                # only some channels
                wfs_chan = wfs_chan[:, :, np.in1d(wf_chans, local_chan_inds)]
                wfs.append(wfs_chan)

            # put noise to enhance clusters
            wfs.append(noise[:, :, local_chan_inds])
            wfs = np.concatenate(wfs, axis=0)
            local_peak_ind = np.concatenate(local_peak_ind, axis=0)
            t1 = time.perf_counter()
            print('WFS time',  t1 - t0)
            

            # reduce dim : PCA
            t0 = time.perf_counter()
            n = d['n_components_by_channel']
            local_feature = np.zeros((wfs.shape[0], d['n_components_by_channel'] * len(local_chan_inds)))

            #~ tsvd = sklearn.decomposition.TruncatedSVD(n_components=n)
            #~ plot_labels = []
            #~ for c in range(wfs.shape[2]):
                #~ local_feature[:, c*n:(c+1)*n] = tsvd.fit_transform(wfs[:, :, c])
            #~ pca = sklearn.decomposition.PCA(n_components=d['n_components_by_channel'], whiten=True)
            #~ local_feature = pca.fit_transform(local_feature)

            pca = sklearn.decomposition.TruncatedSVD(n_components=n)
            for c, chan_ind in enumerate(local_chan_inds):
                local_feature[:, c*n:(c+1)*n] = pca.fit_transform(wfs[:, :, c])

            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            t1 = time.perf_counter()
            print('PCA time',  t1 - t0)

            
            # find some clusters
            t0 = time.perf_counter()
            clusterer = hdbscan.HDBSCAN(min_cluster_size=d['min_cluster_size'], allow_single_cluster=True, metric='l2')
            all_labels = clusterer.fit_predict(local_feature)
            t1 = time.perf_counter()
            print('HDBSCAN time',  t1 - t0)
            

            t0 = time.perf_counter()
            local_labels = all_labels[:-noise.shape[0]]
            noise_labels = all_labels[-noise.shape[0]:]
            
            local_labels_set = np.unique(local_labels)
            
            num_cluster = np.sum(local_labels_set>=0)
            
            if num_cluster > 1:
                # take only the best cluster = best amplitude on central channel
                # other cluster will be taken in a next loop
                ind = local_chan_inds.tolist().index(local_chan_ind)
                peak_values = wfs[:-noise.shape[0], nbefore, ind]
                peak_values = np.abs(peak_values)
                label_peak_values = np.zeros(local_labels_set.size)
                for l, label in enumerate(local_labels_set):
                    if label == -1:
                        continue
                    mask = local_labels == label
                    label_peak_values[l] = np.mean(peak_values[mask])
                best_label = local_labels_set[np.argmax(label_peak_values)]
                final_peak_inds = local_peak_ind[local_labels == best_label]

                # trash outliers from this channel (propably some collision)
                outlier_inds, = np.nonzero((local_labels == -1) & (peaks[local_peak_ind]['channel_ind'] == local_chan_ind))
                if outlier_inds.size > 0:
                    peak_labels[local_peak_ind[outlier_inds]] = -1
            elif num_cluster == 1:
                best_label = 0
                final_peak_inds = local_peak_ind[local_labels >= 0]
            else:
                best_label = None
                final_peak_inds = np.array([], dtype='int64')
                # trash all peaks from this channel
                to_trash_ind,  = np.nonzero(peaks[local_peak_ind]['channel_ind'] == local_chan_ind)
                peak_labels[local_peak_ind[to_trash_ind]] = -1
                
            if best_label is not None:
                if final_peak_inds.size >= d['min_cluster_size']:
                    peak_labels[final_peak_inds] = actual_label
                else:
                    peak_labels[final_peak_inds] = -actual_label
                actual_label += 1

            t1 = time.perf_counter()
            print('label time',  t1 - t0)
            
            # this force recompute amplitude and count at next loop
            prev_local_chan_inds = local_chan_inds

            
            # DEBUG plot
            #~ plot_debug = True
            plot_debug = False
            
            if plot_debug:
                import matplotlib.pyplot as plt
                import umap

                reducer = umap.UMAP()
                reduce_local_feature_all = reducer.fit_transform(local_feature)
                reduce_local_feature = reduce_local_feature_all[:-noise.shape[0]]
                reduce_local_feature_noise = reduce_local_feature_all[-noise.shape[0]:]
                
                wfs_no_noise = wfs[:-noise.shape[0]]
                
                fig, axs = plt.subplots(ncols=3)
                cmap = plt.get_cmap('jet', np.unique(local_labels).size)
                cmap = { label: cmap(l) for l, label in enumerate(local_labels_set) }
                cmap[-1] = 'k'
                for label in local_labels_set:
                    color = cmap[label]
                    ax = axs[0]
                    mask = (local_labels == label)
                    ax.scatter(reduce_local_feature[mask, 0], reduce_local_feature[mask, 1], color=color)
                    
                    # scatter noise
                    mask_noise = (noise_labels == label)
                    if np.any(mask_noise):
                        ax.scatter(reduce_local_feature_noise[mask_noise, 0], reduce_local_feature_noise[mask_noise, 1], color=color, marker='*')
                        
                    ax = axs[1]
                    wfs_flat2 = wfs_no_noise[mask, :, :].swapaxes(1, 2).reshape(np.sum(mask), -1).T
                    ax.plot(wfs_flat2, color=color)
                    if label == best_label:
                        ax.plot(np.mean(wfs_flat2, axis=1), color='m', lw=2)
                    if num_cluster > 1:
                        if outlier_inds.size > 0:
                            wfs_flat2 = wfs_no_noise[outlier_inds, :, :].swapaxes(1, 2).reshape(outlier_inds.size, -1).T
                            ax.plot(wfs_flat2, color='red', ls='--')
                    if num_cluster > 1:
                        ax = axs[2]
                        count, bins = np.histogram(peak_values[mask], bins=35)
                        ax.plot(bins[:-1], count, color=color)
                ax = axs[1]
                for c in range(len(local_chan_inds)):
                    ax.axvline(c * (nbefore + nafter) + nbefore, color='k', ls='--')
                ax.set_title(f'n={local_peak_ind.size} labeled={final_peak_inds.size} chans={local_chan_ind} {local_chan_inds}')

                ax = axs[2]
                sel, = np.nonzero((peaks['channel_ind'] == local_chan_ind) & (peak_labels == 0))
                count, bins = np.histogram(np.abs(peaks['amplitude'][sel]), bins=200)
                ax.plot(bins[:-1], count, color='k', alpha=0.5)

                plt.show()
            # END DEBUG plot
        
        peak_labels[peak_labels == 0] = -1
        
        return peak_labels

    @classmethod
    def _prepare_clean(cls, recording, peaks, wfs_arrays, sparsity_mask, peak_labels, d):
        
        tmp_folder = d['tmp_folder']
        if tmp_folder is None:
            wf_folder = None
        else:
            wf_folder = tmp_folder / 'waveforms_pre_clean'
            wf_folder.mkdir()
        
        
        num_chans = recording.get_num_channels()
        fs = recording.get_sampling_frequency()
        nbefore = int(d['ms_before'] * fs / 1000.)
        nafter = int(d['ms_after'] * fs / 1000.)
        
        possible_channel_inds = np.unique(peaks['channel_ind'])
        chan_distances = get_channel_distances(recording)
        closest_channels = []
        for c in range(num_chans):
            chans, = np.nonzero(chan_distances[c, :] <= ( d['radius_um']) * 2)
            closest_channels.append(chans)

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]
        
        # loop over label take wafevorm from channel and get main channel
        main_channels = []
        for l, label in enumerate(labels):
            wfs, chan_inds = _collect_sparse_waveforms(peaks, wfs_arrays, closest_channels, peak_labels, sparsity_mask, label)
            template = np.mean(wfs, axis=0)
            main_chan = chan_inds[np.argmax(np.max(np.abs(template), axis=0))]
            main_channels.append(main_chan)
        
        # extact again waveforms based on new sparsity mask depending on main_chan
        dtype = recording.get_dtype()
        #~ return_scaled = False
        peak_dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
        keep = peak_labels>=0
        num_keep = np.sum(keep)
        keep_peak_labels = peak_labels[keep]
        peaks2 = np.zeros(num_keep, dtype=peak_dtype)
        peaks2['sample_ind'] = peaks['sample_ind'][keep]
        peaks2['segment_ind'] = peaks['segment_ind'][keep]
        sparsity_mask2 = np.zeros((labels.shape[0], num_chans), dtype='bool')
        for l, label in enumerate(labels):
            main_chan = main_channels[l]
            mask = keep_peak_labels == label
            peaks2['unit_ind'][mask] = l
            # here we take a twice radius
            closest_chans, = np.nonzero(chan_distances[main_chan, :] <= d['radius_um'] * 2)
            sparsity_mask2[l, closest_chans] = True

        wfs_arrays2 = extract_waveforms_to_buffers(recording, peaks2, labels, nbefore, nafter,
                                mode=d['waveform_mode'], return_scaled=False, folder=wf_folder, dtype=recording.get_dtype(),
                                sparsity_mask=sparsity_mask2,  copy=(d['waveform_mode'] == 'shared_memory'),
                                **d['job_kwargs'])
        
        return wfs_arrays2, sparsity_mask2
    
    @classmethod
    def _clean_cluster(cls, recording, peaks, wfs_arrays2, sparsity_mask2, peak_labels, d):

        fs = recording.get_sampling_frequency()
        nbefore = int(d['ms_before'] * fs / 1000.)
        nafter = int(d['ms_after'] * fs / 1000.)

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        chan_locs = recording.get_channel_locations()
        channel_distances = get_channel_distances(recording)
        
        clean_peak_labels, peak_sample_shifts = auto_clean_clustering(wfs_arrays2, sparsity_mask2, labels, peak_labels, nbefore, nafter, channel_distances,
                                radius_um=d['radius_um'], auto_merge_num_shift=d['auto_merge_num_shift'], auto_merge_quantile_limit=d['auto_merge_quantile_limit'])
        
        return  clean_peak_labels, peak_sample_shifts
        



def _collect_sparse_waveforms(peaks, wfs_arrays, closest_channels, peak_labels, sparsity_mask, label):
    inds, = np.nonzero(peak_labels == label)
    local_peaks = peaks[inds]
    label_chan_inds, count = np.unique(local_peaks['channel_ind'], return_counts=True)
    main_chan = label_chan_inds[np.argmax(count)]

    #only main channel sparsity
    wanted_chans = closest_channels[main_chan]
    for chan_ind in label_chan_inds:
        # remove channel non in common
        wanted_chans = np.intersect1d(wanted_chans, closest_channels[chan_ind])
    # print('wanted_chans', wanted_chans)


    wfs = []
    for chan_ind in label_chan_inds:
        sel,  = np.nonzero(peaks['channel_ind'] == chan_ind)

        inds,  = np.nonzero(peak_labels[sel] == label)
        

        wf_chans, = np.nonzero(sparsity_mask[chan_ind])
        # print('wf_chans', wf_chans)
        # TODO: only for debug, remove later
        assert np.all(np.in1d(wanted_chans, wf_chans))
        wfs_chan = wfs_arrays[chan_ind]

        # TODO: only for debug, remove later
        assert wfs_chan.shape[0] == sel.size

        
        wfs_chan = wfs_chan[inds, :, :]
        # only some channels
        wfs_chan = wfs_chan[:, :, np.in1d(wf_chans, wanted_chans)]
        wfs.append(wfs_chan)
    
    wfs = np.concatenate(wfs, axis=0)
    
    # TODO DEBUG and check
    assert wanted_chans.shape[0] == wfs.shape[2]
    
    return wfs, wanted_chans






class PositionAndPCAClustering:
    """
    Stupid clustering.
    peak are clustered from there channel detection
    So peak['channel_ind'] will be the peak_labels
    """
    """
    hdbscan clustering

    The idea is to combine the positions of the cells, as obtained with
    a chosen localization method, and some key features such as ptp/mean/std
    to get a rather robust and quick clustering

    peak are clustered from there channel detection
    So peak['channel_ind'] will be the peak_labels
    
    """
    _default_params = {
        "peak_locations" : None,
        'ms_before': 1.5,
        'ms_after': 2.5,
        "n_components_by_channel" : 3,
        "job_kwargs" : {"n_jobs" : -1, "chunk_memory" : "10M", "progress_bar" : True},
        "hdbscan_params": {"min_cluster_size" : 50, "probability_thr" : 0, "allow_single_cluster" : True},
        "debug" : True,
        "local_radius_um" : 100,
        "tmp_folder" : None
    }

    @classmethod
    def launch_clustering(cls, to_cluster_from, probability_thr, clustering_params):

        import hdbscan
        clustering = hdbscan.hdbscan(to_cluster_from, **clustering_params)
        peak_labels = clustering[0]
        peak_labels_persistent = peak_labels.copy()
        cluster_probability = clustering[2]

        persistent_clusters, = np.nonzero(clustering[2] > probability_thr)
        mask = np.in1d(peak_labels, persistent_clusters)

        peak_labels_persistent[~mask] = -2
        
        return peak_labels_persistent

    @classmethod
    def plot_clusters(cls, data, labels, plot_labels, output):
        import pylab as plt
        nb_dims = data.shape[1]

        fig, ax = plt.subplots(nb_dims, nb_dims)

        for i in range(nb_dims):
            for j in range(nb_dims):
                ax[i, j].spines['top'].set_visible(False)
                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if j > i:
                    ax[i, j].scatter(data[:, i], data[:, j], c=labels, s=1, alpha=0.1)

                    for l in np.unique(labels):
                        mask = labels == l
                        x, y = np.mean(data[mask, i]), np.mean(data[mask, j])
                        ax[i,j].text(x, y, f'{l}')

                else:
                    ax[i, j].axis("off")


        for i in range(nb_dims):
            ax[i, 0].set_ylabel(plot_labels[i])

        for j in range(nb_dims):
            ax[nb_dims-1, i].set_xlabel(plot_labels[i])

        plt.tight_layout()

        #plt.show()
        plt.savefig(output)
        plt.close()


    #@classmethod
    #def main_function(cls, recording, peaks, params):



    @classmethod
    def main_function(cls, recording, peaks, params):

        assert params['peak_locations'] is not None, "Peak locations should not be None!"

        fs = recording.get_sampling_frequency()
        nbefore = int(params['ms_before'] * fs / 1000.)
        nafter = int(params['ms_after'] * fs / 1000.)

        tmp_folder = params['tmp_folder']
        if tmp_folder is None:
            wf_folder = None
        else:
            wf_folder = Path(tmp_folder) / 'waveforms_clustering'
            wf_folder.mkdir(parents=True)

        clustering_params = {}
        clustering_params.update(params['hdbscan_params'])
        clustering_params['core_dist_n_jobs'] = params['job_kwargs']['n_jobs']
        probability_thr = clustering_params.pop('probability_thr')

        if params['debug']:
            tmp_folder = Path(params['tmp_folder'])
            tmp_folder.mkdir(exist_ok=True)

        positions = params['peak_locations']
        
        if params['debug']:
            import pylab as plt
            import spikeinterface.full as si

            si.plot_probe_map(recording)
            plt.scatter(positions['x'], positions['y'], alpha=0.1, s=1)

            plt.savefig(tmp_folder / 'positions.png')
            plt.close()

        plot_labels = []

        to_cluster_from = []
        for key in ['x', 'y']: ##Think about adding z ?
            to_cluster_from += [positions[key]]
            plot_labels += [key]

        to_cluster_from = np.stack(to_cluster_from, axis=1)

        pre_labels = cls.launch_clustering(to_cluster_from, probability_thr, clustering_params)

        if params['debug'] is not None:
            filename = tmp_folder / 'clustering.png'
            cls.plot_clusters(to_cluster_from, pre_labels, plot_labels, filename)

        # extact again waveforms based on new sparsity mask depending on main_chan
        dtype = recording.get_dtype()
        return_scaled = False
        peak_dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
        keep = pre_labels >= 0
    
        num_keep = np.sum(keep)
        keep_peak_labels = pre_labels[keep]
    
        labels = np.unique(keep_peak_labels)

        peaks2 = np.zeros(num_keep, dtype=peak_dtype)
        peaks2['sample_ind'] = peaks['sample_ind'][keep]
        peaks2['segment_ind'] = peaks['segment_ind'][keep]

        num_chans = recording.get_num_channels()
        sparsity_mask = np.zeros((labels.shape[0], num_chans), dtype='bool')

        chan_locs = recording.get_channel_locations()
        chan_distances = get_channel_distances(recording)
        
        print(np.unique(labels))
        for l, label in enumerate(labels):
            mask = keep_peak_labels == label
            peaks2['unit_ind'][mask] = l

            x, y = np.mean(positions['x'][keep][mask]), np.mean(positions['y'][keep][mask])
            main_chan = np.argmin(np.linalg.norm(chan_locs - np.array([[x, y]]), axis=1))
            print(x, y, main_chan)

            # here we take a biger radius
            closest_chans, = np.nonzero(chan_distances[main_chan, :] <= params['local_radius_um'])
            sparsity_mask[l, closest_chans] = True

        wfs_arrays, wfs_arrays_info = allocate_waveforms(recording, peaks2, labels, nbefore, nafter,
                                                         mode='memmap', folder=wf_folder, dtype=dtype, sparsity_mask=sparsity_mask)
        distribute_waveforms_to_buffers(recording, peaks2,  labels, wfs_arrays_info, nbefore, nafter, return_scaled,
                                        mode='memmap', sparsity_mask=sparsity_mask, **params['job_kwargs'])

        peak_labels_ = -1 * np.ones(num_keep, dtype=np.int64)
        nb_clusters = 0
        main_channels = {}
        for l in range(len(labels)):
            label = labels[l]
            mask, = np.nonzero(keep_peak_labels == label)
            wfs = wfs_arrays[label]

            n = params['n_components_by_channel']
            local_feature = np.zeros((wfs.shape[0], n * wfs.shape[2]))
            tsvd = sklearn.decomposition.TruncatedSVD(n_components=n)
            plot_labels = []
            for c in range(wfs.shape[2]):
                local_feature[:, c*n:(c+1)*n] = tsvd.fit_transform(wfs[:, :, c])
            
            pca = sklearn.decomposition.PCA(n_components=5, whiten=True)
            local_feature = pca.fit_transform(local_feature)


            plot_labels = np.arange(local_feature.shape[1])
            local_labels = cls.launch_clustering(local_feature, 0, clustering_params)

            if params['debug'] is not None:

                local_labels_2 = local_labels.copy()
                local_labels_2[local_labels_2 >= 0] += nb_clusters + 1
                filename = tmp_folder / f'local_clustering_{label}.png'
                cls.plot_clusters(local_feature, local_labels_2, plot_labels, filename)

            mask2, = np.nonzero(local_labels >= 0)

            peak_labels_[mask[mask2]] = local_labels[mask2] + nb_clusters

            for label in np.unique(local_labels[mask2]):
                template = np.mean(wfs[local_labels == label, :, :], axis=0)
                ind_max = np.argmax(np.max(np.abs(template), axis=0))
                chans, = np.nonzero(sparsity_mask[l, :])
                main_channels[label + nb_clusters] = chans[ind_max]

            nb_clusters += local_labels.max() + 1
            print(nb_clusters)


        peak_labels = -2 * np.ones(peaks.size, dtype=np.int64)
        peak_labels[keep] = peak_labels_

        to_cluster_from = np.stack([positions['x'], positions['y']], axis=1)[keep]
        cls.plot_clusters(to_cluster_from, peak_labels_, ['x', 'y'], 'global_clustering.png')


        # clean
        # keep_clean = (peak_labels >=0)
        # peak_labels_clean =peak_labels[keep_clean] 
        # labels = np.unique(peak_labels_clean)
        
        # wf_folder_clean = Path(tmp_folder) / 'waveforms_clustering_clean'
        # wf_folder_clean.mkdir(parents=True)
        
        # n = np.sum(keep_clean)
        # sparsity_mask2 = np.zeros((n, num_chans), dtype='bool')
        # peaks3 = np.zeros(n, dtype=peak_dtype)
        # peaks3['sample_ind'] = peaks['sample_ind'][keep_clean]
        
        # peaks3['segment_ind'] = peaks['segment_ind'][keep_clean]
        # for l, label in enumerate(labels):
        #     main_chan = main_channels[label]
        #     closest_chans, = np.nonzero(chan_distances[main_chan, :] <= params['local_radius_um'])
        #     sparsity_mask2[l, closest_chans] = True
        #     peaks3['unit_ind'][peak_labels_clean == label] = l
        #     print(labels, closest_chans, main_chan)

        # wfs_arrays2, wfs_arrays_info2 = allocate_waveforms(recording, peaks3, labels, nbefore, nafter,
        #                                                  mode='memmap', folder=wf_folder_clean,
        #                                                  dtype=dtype, sparsity_mask=sparsity_mask2)
        # distribute_waveforms_to_buffers(recording, peaks3,  labels, wfs_arrays_info2, nbefore, nafter, return_scaled,
        #                                 mode='memmap', sparsity_mask=sparsity_mask2, **params['job_kwargs'])
        # print(wfs_arrays.keys())


        # main_channels = []
        # templates = np.zeros((labels.size, nbefore+nafter, num_chans), dtype=np.float32)

        # for l, label in enumerate(labels):
        #     wfs = wfs_arrays2[label]
        #     template = np.mean(wfs, axis=0)
        #     chan_inds, = np.nonzero(sparsity_mask2[l, :])
        #     templates[l, :, chan_inds] =  template
        #     assert wfs.shape[2] == chan_inds.size
        #     main_chan = chan_inds[np.argmax(np.max(np.abs(template), axis=0))]
        #     templates.append(template)
        #     main_channels.append(main_chan)

        # print(main_channels)
        # ## Step 1 : auto merge
        # # auto_merge_list = []
        # auto_trash_list = []
        # for l0 in  range(len(labels)):
        #     for l1 in  range(l0+1, len(labels)):
                
        #         label0, label1 = labels[l0], labels[l1]

        #         main_chan0 = main_channels[l0]
        #         main_chan1 = main_channels[l1]
                
        #         channel_inds0, = np.nonzero(sparsity_mask2[l0, :])
        #         channel_inds1, = np.nonzero(sparsity_mask2[l1, :])

                
        #         intersect_chans = np.intersect1d(channel_inds0, channel_inds1)
        #         if len(intersect_chans) == 0:
        #             continue
                 
        #         wfs0 = wfs_arrays2[label0]
        #         wfs0 = wfs0[:, :,np.in1d(channel_inds0, intersect_chans)]
        #         wfs1 = wfs_arrays2[label1]
        #         wfs1 = wfs1[:, :,np.in1d(channel_inds1, intersect_chans)]
                
        #         # TODO : remove
        #         assert wfs0.shape[2] == wfs1.shape[2]
                
        #         auto_merge_num_shift = 7
        #         auto_merge_quantile_limit = 0.8
        #         equal, shift = check_equal_template_with_distribution_overlap(wfs0, wfs1,
        #                         num_shift=auto_merge_num_shift, quantile_limit=auto_merge_quantile_limit, 
        #                         return_shift=True, debug=(label0 == 9) and (label1 == 17))
                
        #         print(label0, label1, equal)
        #         if equal:
        #             if shift == 0:
        #                 if wfs0.shape[0] >= wfs1.shape[0]:
        #                     auto_trash_list.append(l1)
        #                 else:
        #                     auto_trash_list.append(l0)
        #             else:
        #                 ind0 = list(channel_inds0).index(main_chan0)
        #                 ind_sample0 = np.argmax(np.abs(templates[l0][:, ind0]))
        #                 ind1 = list(channel_inds1).index(main_chan1)
        #                 ind_sample1 = np.argmax(np.abs(templates[l1][:, ind1]))
        #                 best = np.argmin(np.abs(np.array([ind_sample0, ind_sample1]) - nbefore))
        #                 if best == 0:
        #                     auto_trash_list.append(l1)
        #                 else:
        #                     auto_trash_list.append(l0)

        #         # DEBUG plot
        #         #~ plot_debug = debug
        #         #plot_debug = True
        #         plot_debug = False
        #         # plot_debug = equal
        #         plot_debug = (label0 == 9) and (label1 == 17)
        #         if plot_debug :
        #             import matplotlib.pyplot as plt
        #             wfs_flat0 = wfs0.swapaxes(1, 2).reshape(wfs0.shape[0], -1).T
        #             wfs_flat1 = wfs1.swapaxes(1, 2).reshape(wfs1.shape[0], -1).T
        #             fig, ax = plt.subplots()
        #             ax.plot(wfs_flat0, color='g', alpha=0.1)
        #             ax.plot(wfs_flat1, color='r', alpha=0.1)

        #             ax.plot(np.mean(wfs_flat0, axis=1), color='c', lw=2)
        #             ax.plot(np.mean(wfs_flat1, axis=1), color='m', lw=2)
                    
        #             for c in range(len(intersect_chans)):
        #                 ax.axvline(c * (nbefore + nafter) + nbefore, color='k', ls='--')
        #             ax.set_title(f'label0={label0} label1={label1} equal{equal} shift{shift}')
        #             plt.show()
        
        # print('auto_trash_list', auto_trash_list)


        # peak_labels[np.in1d(peak_labels, auto_trash_list)] = -3



        return np.unique(peak_labels), peak_labels



clustering_methods = {
    'stupid' : StupidClustering,
    'position_clustering' : PositionClustering,
    'sliding_hdbscan': SlidingHdbscanClustering,
    'position_pca_clustering' : PositionAndPCAClustering
}

