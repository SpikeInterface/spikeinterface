"""Sorting components: clustering"""
import random
import string
from pathlib import Path

import numpy as np

import sklearn.decomposition

import hdbscan
import umap

from ..core import get_global_tmp_folder, extract_waveforms, NumpySorting
from ..toolkit import get_channel_distances


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
    
    """
    _default_params = {
        'tmp_folder': None,
        'ms_before': 1.5,
        'ms_after': 2.5,
        'max_spikes_per_channel': 1000,
        'min_spike_on_channel' : 5,
        'min_cluster_size' : 10,
        'radius_um': 50.,
        'n_components_by_channel': 4,
        'job_kwargs' : {}
    }

    @classmethod
    def _initlialize_folder(cls, recording, peaks, params):
        print('_initlialize_folder')
        d = params

        tmp_folder = params['tmp_folder']
        
        if tmp_folder is None:
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = get_global_tmp_folder() / f'SlidingHdbscanClustering_{name}'
        else:
            tmp_folder = Path(tmp_folder)
        
        possible_chans, count = np.unique(peaks['channel_ind'], return_counts=True)
        print(possible_chans)
        print(count)
        
        
        sorting = NumpySorting.from_times_labels(peaks['sample_ind'], peaks['channel_ind'], recording.get_sampling_frequency())
        sorting = sorting.save(folder=tmp_folder / 'by_channel_peaks')
        print([sorting.get_unit_spike_train(unit_id).size for unit_id in sorting.unit_ids])
        
        
        we = extract_waveforms(recording, sorting, 
                    tmp_folder / 'by_chan_waveform',
                    load_if_exists=False,
                    precompute_template=[],
                    ms_before=d['ms_before'], ms_after=d['ms_after'],
                    #~ max_spikes_per_unit=d['max_spikes_per_channel'],
                    max_spikes_per_unit=None,
                    overwrite=False,
                    return_scaled=False,
                    dtype=None,
                    use_relative_path=False,
                    **d['job_kwargs'])
        
        return we

    @classmethod
    def _find_clusters(cls, recording, peaks, we, d):
        
        
        num_chans = recording.get_num_channels()
        #~ channel_explored = np.zeros(num_chans, dtype='bool')
        
        possible_inds = we.sorting.unit_ids
        chan_distances = get_channel_distances(recording)
        closest_channels = []
        for c in range(num_chans):
            chans, = np.nonzero(chan_distances[c, :] <= d['radius_um'])
            chans = np.intersect1d(possible_inds, chans)
            closest_channels.append(chans)
        
        
        possible_chan_inds = we.sorting.unit_ids
        #~ print('possible_chan_inds', possible_chan_inds)
        #
        peak_labels = np.zeros(peaks.size, dtype='int64')
        
        
        actual_label = 1
        
        while True:
            #~ print('*'*10)
            count = [np.sum(peaks['channel_ind'] == chan_ind) for chan_ind in possible_chan_inds]
            count2 = [np.sum((peaks['channel_ind'] == chan_ind) & (peak_labels==0)) for chan_ind in possible_chan_inds]
            #~ print(peaks.size, peak_labels.size, np.sum(count))
            #~ print('possible_chan_inds', possible_chan_inds)
            #~ print(count)
            #~ print(count2)
            
            #~ print('Not label', np.sum(peak_labels==0), peak_labels.size)
            
            # get best channel
            chan_amps = np.zeros(num_chans)
            not_label_mask = (peak_labels == 0)
            for c, chan_ind in enumerate(possible_chan_inds):
                #~ if channel_explored[c]:
                    #~ continue
                sel, = np.nonzero((peaks['channel_ind'] == chan_ind) & not_label_mask)
                
                #~ print('chan_ind', chan_ind, 'sel', sel.size)
                #~ sel2, = np.nonzero(peak_labels[sel] == 0)
                amps = np.abs(peaks[sel]['amplitude'])
                #~ print(sel.size, amps.size, amps.size <= d['min_cluster_size'])
                if amps.size <= d['min_spike_on_channel']:
                    # TODO should relax this because
                    # cluster could be spread over several channel
                    continue
                chan_amps[chan_ind] = np.percentile(amps, 90)
            #~ print('chan_amps', chan_amps)
            if np.all(chan_amps == 0):
                break
            local_chan_ind = np.argmax(chan_amps)
            local_chan_inds = closest_channels[local_chan_ind]

            print('local_chan_ind', local_chan_ind)
            
            # take waveforms not label yet for channel in radius
            wfs = []
            local_peak_ind = []
            for chan_ind in local_chan_inds:
                sel,  = np.nonzero(peaks['channel_ind'] == chan_ind)
                inds,  = np.nonzero(peak_labels[sel] == 0)
                #~ print(inds)
                
                local_peak_ind.append(sel[inds])
                
                # here a unit is a channel index!!!
                wfs_chan = we.get_waveforms(chan_ind, with_index=False, cache=False, memmap=True, sparsity=None)
                #~ print(chan_ind, 'wfs_chan.shape', wfs_chan.shape, sel.size, inds.size)
                assert wfs_chan.shape[0] == sel.size
                
                #~ print(wfs_chan.shape)
                wfs_chan = wfs_chan[inds, :, :][:, :, local_chan_inds]
                wfs.append(wfs_chan)
            wfs = np.concatenate(wfs, axis=0)
            local_peak_ind = np.concatenate(local_peak_ind, axis=0)
            #~ print('wfs.shape', wfs.shape, 'local_peak_ind.size', local_peak_ind.size)
            #~ assert local_peak_ind.size == wfs.shape[0]

            # reduce dim : PCA + UMAP
            n = d['n_components_by_channel']
            local_feature = np.zeros((wfs.shape[0], d['n_components_by_channel'] * len(local_chan_inds)))
            pca = sklearn.decomposition.TruncatedSVD(n_components=n)
            for c, chan_ind in enumerate(local_chan_inds):
                local_feature[:, c*n:(c+1)*n] = pca.fit_transform(wfs[:, :, c])
            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            
            #~ reducer = umap.UMAP()
            #~ local_feature = reducer.fit_transform(local_feature)

            #~ wfs_flat = wfs.reshape(wfs.shape[0], -1)
            #~ reducer = umap.UMAP()
            #~ local_feature = reducer.fit_transform(wfs_flat)
            
            # find some clusters
            clusterer = hdbscan.HDBSCAN(min_cluster_size=d['min_cluster_size'], allow_single_cluster=True, metric='l2')
            local_labels = clusterer.fit_predict(local_feature)
            
            local_labels_set = np.unique(local_labels)
            
            several_cluster = np.sum(local_labels_set>=0) > 1
            
            print('several_cluster', several_cluster)
            
            if several_cluster:
                # take only the best cluster = best amplitude on central channel
                # other cluster will be taken in a next loop
                ind = local_chan_inds.tolist().index(local_chan_ind)
                peak_values = wfs[:, we.nbefore, ind]
                peak_values = np.abs(peak_values)
                label_peak_values = np.zeros(local_labels_set.size)
                for l, label in enumerate(local_labels_set):
                    if label == -1:
                        continue
                    mask = local_labels == label
                    label_peak_values[l] = np.mean(peak_values[mask])
                best_label = local_labels_set[np.argmax(label_peak_values)]
                #~ assert local_peak_ind.size == local_labels.size
                final_peak_inds = local_peak_ind[local_labels == best_label]
                
                outlier_inds, = np.nonzero((local_labels == -1) & (peaks[local_peak_ind]['channel_ind'] == local_chan_ind))
                if outlier_inds.size > 0:
                    peak_labels[outlier_inds] = -1
            else:
                # TODO: find more strict criteria
                final_peak_inds = local_peak_ind[local_labels >= 0]
            
            peak_labels[final_peak_inds] = actual_label
            actual_label += 1


                
            
            

            # DEBUG plot
            #~ plot_debug = local_chan_ind in(21, 22)
            #~ plot_debug = True
            #~ plot_debug = False
            #~ plot_debug = actual_label >= 15
            plot_debug = not several_cluster
            
            
            if plot_debug:
                import matplotlib.pyplot as plt

                reducer = umap.UMAP()
                reduce_local_feature = reducer.fit_transform(local_feature)                
                
                fig, axs = plt.subplots(ncols=3)
                cmap = plt.get_cmap('jet', np.unique(local_labels).size)
                cmap = { label: cmap(l) for l, label in enumerate(local_labels_set) }
                cmap[-1] = 'k'
                for label in local_labels_set:
                    color = cmap[label]
                    ax = axs[0]
                    mask = (local_labels == label)
                    #~ ax.scatter(local_feature[mask, 0], local_feature[mask, 1], color=color)
                    ax.scatter(reduce_local_feature[mask, 0], reduce_local_feature[mask, 1], color=color)
                    
                    ax = axs[1]
                    wfs_flat2 = wfs[mask, :, :].swapaxes(1, 2).reshape(np.sum(mask), -1).T
                    ax.plot(wfs_flat2, color=color)
                    if label == best_label:
                        ax.plot(np.mean(wfs_flat2, axis=1), color='m', lw=2)
                    if outlier_inds.size > 0:
                        wfs_flat2 = wfs[outlier_inds, :, :].swapaxes(1, 2).reshape(outlier_inds.size, -1).T
                        ax.plot(wfs_flat2, color='red', ls='--')
                    if several_cluster:
                        ax = axs[2]
                        count, bins = np.histogram(peak_values[mask], bins=35)
                        ax.plot(bins[:-1], count, color=color)
                ax = axs[1]
                for c in range(len(local_chan_inds)):
                    ax.axvline(c * (we.nbefore + we.nafter) + we.nbefore, color='k', ls='--')
                ax.set_title(f'n={local_peak_ind.size} labeled={final_peak_inds.size} chans={local_chan_ind} {local_chan_inds}')
                plt.show()
            # END DEBUG plot
        
        peak_labels[peak_labels == 0] = -1
        
        return peak_labels
    
    @classmethod
    def main_function(cls, recording, peaks, params):
        we = cls._initlialize_folder(recording, peaks, params)
        peak_labels = cls._find_clusters(recording, we, params)
        
        
        #~ tmp_folder = params['tmp_folder']
        
        #~ if tmp_folder is None:
            #~ name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            #~ tmp_folder = get_global_tmp_folder() / f'SlidingHdbscanClustering_{name}'
        
        
        
        #~ labels = np.arange(recording.get_num_channels(), dtype='int64')
        #~ peak_labels = peaks['channel_ind']
        labels, peak_labels = None, None
        return labels, peak_labels
    




clustering_methods = {
    'stupid' : StupidClustering,
    'sliding_hdbscan': SlidingHdbscanClustering,
}

