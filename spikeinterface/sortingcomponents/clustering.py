"""Sorting components: clustering"""
import random
import string
from pathlib import Path

import numpy as np

import sklearn.decomposition

try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False


from ..core import get_global_tmp_folder
from ..toolkit import get_channel_distances, get_random_data_chunks
from ..core.waveform_tools import allocate_waveforms, distribute_waveforms_to_buffers




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
        'tmp_folder': None,
        'ms_before': 1.5,
        'ms_after': 2.5,
        'noise_size' : 300,
        'min_spike_on_channel' : 5,
        'stop_explore_percent' : 0.05,
        'min_cluster_size' : 10,
        'radius_um': 50.,
        'n_components_by_channel': 4,
        'job_kwargs' : {}
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, 'SlidingHdbscanClustering need hdbscan to be installed'
        we = cls._initlialize_folder(recording, peaks, params)
        peak_labels = cls._find_clusters(recording, peaks, we, params)
        labels = np.unique(peak_labels)
        labels, peak_labels = None, None
        return labels, peak_labels

    @classmethod
    def _initlialize_folder(cls, recording, peaks, params):
        d = params
        tmp_folder = params['tmp_folder']
        if tmp_folder is None:
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = get_global_tmp_folder() / f'SlidingHdbscanClustering_{name}'
        else:
            tmp_folder = Path(tmp_folder)
        tmp_folder.mkdir()
        
        # create a new peak vector to extract waveforms
        dtype = [('sample_ind', 'int64'), ('unit_ind', 'int64'), ('segment_ind', 'int64')]
        peaks2 = np.zeros(peaks.size, dtype=dtype)
        peaks2['sample_ind'] = peaks['sample_ind']
        peaks2['unit_ind'] = peaks['channel_ind']
        peaks2['segment_ind'] = peaks['segment_ind']
        
        unit_ids = recording.channel_ids
        fs = recording.get_sampling_frequency()
        dtype = recording.get_dtype()
        
        nbefore = int(d['ms_before'] * fs / 1000.)
        nafter = int(d['ms_after'] * fs / 1000.)
        
        return_scaled = False
        wfs_arrays, wfs_arrays_info = allocate_waveforms(recording, peaks2, unit_ids, nbefore, nafter, mode='memmap', folder=tmp_folder, dtype=dtype)
        distribute_waveforms_to_buffers(recording, peaks2, unit_ids, wfs_arrays_info, nbefore, nafter, return_scaled, **d['job_kwargs'])
        
        return wfs_arrays, wfs_arrays_info
        
        
        #~ sorting = NumpySorting.from_times_labels(peaks['sample_ind'], peaks['channel_ind'], recording.get_sampling_frequency())
        #~ sorting = sorting.save(folder=tmp_folder / 'by_channel_peaks')
        
        #~ we = extract_waveforms(recording, sorting, 
                    #~ tmp_folder / 'by_chan_waveform',
                    #~ load_if_exists=False,
                    #~ precompute_template=[],
                    #~ ms_before=d['ms_before'], ms_after=d['ms_after'],
                    #~ max_spikes_per_unit=None,
                    #~ overwrite=False,
                    #~ return_scaled=False,
                    #~ dtype=None,
                    #~ use_relative_path=False,
                    #~ **d['job_kwargs'])
        
        #~ return we
    
    @classmethod
    def _find_clusters(cls, recording, peaks, we, d):
        
        num_chans = recording.get_num_channels()
        
        # channel neighborhood
        possible_inds = we.sorting.unit_ids
        chan_distances = get_channel_distances(recording)
        closest_channels = []
        for c in range(num_chans):
            chans, = np.nonzero(chan_distances[c, :] <= d['radius_um'])
            chans = np.intersect1d(possible_inds, chans)
            closest_channels.append(chans)
        
        possible_chan_inds = we.sorting.unit_ids
        peak_labels = np.zeros(peaks.size, dtype='int64')

        noise = get_random_data_chunks(recording, return_scaled=False,
                        num_chunks_per_segment=d['noise_size'], chunk_size=we.nbefore+we.nafter, concatenated=False, seed=None)
        noise = np.stack(noise, axis=0)
        
        
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
            wfs = []
            local_peak_ind = []
            for chan_ind in local_chan_inds:
                sel,  = np.nonzero(peaks['channel_ind'] == chan_ind)
                inds,  = np.nonzero(peak_labels[sel] == 0)
                local_peak_ind.append(sel[inds])
                # here a unit is a channel index!!!
                wfs_chan = we.get_waveforms(chan_ind, with_index=False, cache=False, memmap=True, sparsity=None)
                assert wfs_chan.shape[0] == sel.size
                wfs_chan = wfs_chan[inds, :, :][:, :, local_chan_inds]
                wfs.append(wfs_chan)
            
            # put noise to enhance clusters
            wfs.append(noise[:, :, local_chan_inds])
            wfs = np.concatenate(wfs, axis=0)
            local_peak_ind = np.concatenate(local_peak_ind, axis=0)

            # reduce dim : PCA
            n = d['n_components_by_channel']
            local_feature = np.zeros((wfs.shape[0], d['n_components_by_channel'] * len(local_chan_inds)))
            pca = sklearn.decomposition.TruncatedSVD(n_components=n)
            for c, chan_ind in enumerate(local_chan_inds):
                local_feature[:, c*n:(c+1)*n] = pca.fit_transform(wfs[:, :, c])
            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            
            # find some clusters
            clusterer = hdbscan.HDBSCAN(min_cluster_size=d['min_cluster_size'], allow_single_cluster=True, metric='l2')
            all_labels = clusterer.fit_predict(local_feature)
            
            local_labels = all_labels[:-noise.shape[0]]
            noise_labels = all_labels[-noise.shape[0]:]
            
            local_labels_set = np.unique(local_labels)
            
            num_cluster = np.sum(local_labels_set>=0)
            
            if num_cluster > 1:
                # take only the best cluster = best amplitude on central channel
                # other cluster will be taken in a next loop
                ind = local_chan_inds.tolist().index(local_chan_ind)
                peak_values = wfs[:-noise.shape[0], we.nbefore, ind]
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
                peak_labels[final_peak_inds] = actual_label
                actual_label += 1
            
            # this force recompute amplitude and count at next loop
            prev_local_chan_inds = local_chan_inds

            
            # DEBUG plot
            #~ plot_debug = True
            plot_debug = False
            
            if plot_debug:
                import matplotlib.pyplot as plt

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
                    ax.axvline(c * (we.nbefore + we.nafter) + we.nbefore, color='k', ls='--')
                ax.set_title(f'n={local_peak_ind.size} labeled={final_peak_inds.size} chans={local_chan_ind} {local_chan_inds}')

                ax = axs[2]
                sel, = np.nonzero((peaks['channel_ind'] == local_chan_ind) & (peak_labels == 0))
                count, bins = np.histogram(np.abs(peaks['amplitude'][sel]), bins=200)
                ax.plot(bins[:-1], count, color='k', alpha=0.5)

                plt.show()
            # END DEBUG plot
        
        peak_labels[peak_labels == 0] = -1
        
        return peak_labels
    

    




clustering_methods = {
    'stupid' : StupidClustering,
    'sliding_hdbscan': SlidingHdbscanClustering,
}

