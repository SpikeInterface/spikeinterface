"""
This gather some function usefull for some clusetring algos.
"""

import numpy as np

import torch
import torch.multiprocessing as mp
from scipy.signal import argrelmin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA
import scipy.optimize as optim_ls
import hdbscan

from spikeinterface.sortingcomponents.clustering.isocut5 import isocut5
from scipy.spatial.distance import cdist


from ..toolkit import check_equal_template_with_distribution_overlap



def _split_waveforms(wfs_and_noise, noise_size, nbefore, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug):

    import sklearn.decomposition
    import hdbscan
    
    valid_size = wfs_and_noise.shape[0] - noise_size
    
    local_feature = np.zeros((wfs_and_noise.shape[0], n_components_by_channel * wfs_and_noise.shape[2]))
    tsvd = sklearn.decomposition.TruncatedSVD(n_components=n_components_by_channel)
    for c in range(wfs_and_noise.shape[2]):
        local_feature[:, c*n_components_by_channel:(c+1)*n_components_by_channel] = tsvd.fit_transform(wfs_and_noise[:, :, c])
    n_components = min(n_components, local_feature.shape[1])
    pca = sklearn.decomposition.PCA(n_components=n_components, whiten=True)
    local_feature = pca.fit_transform(local_feature)
    
    # hdbscan on pca
    clustering = hdbscan.hdbscan(local_feature, **hdbscan_params)
    local_labels_with_noise = clustering[0]
    cluster_probability = clustering[2]
    persistent_clusters, = np.nonzero(clustering[2] > probability_thr)
    local_labels_with_noise[~np.in1d(local_labels_with_noise, persistent_clusters)] = -1
    
    # remove super small cluster
    labels, count = np.unique(local_labels_with_noise[:valid_size], return_counts=True)
    mask = labels >= 0
    labels, count = labels[mask], count[mask]
    minimum_cluster_size_ratio = 0.05
    
    #~ print(labels, count)
    
    to_remove = labels[(count / valid_size) <minimum_cluster_size_ratio]
    #~ print('to_remove', to_remove, count / valid_size)
    if to_remove.size > 0:
        local_labels_with_noise[np.in1d(local_labels_with_noise, to_remove)] = -1
    
    local_labels_with_noise[valid_size:] = -2
    

    if debug:
        import matplotlib.pyplot as plt
        import umap
        
        fig, ax = plt.subplots()

        #~ reducer = umap.UMAP()
        #~ local_feature_plot = reducer.fit_transform(local_feature)
        local_feature_plot = local_feature

        
        unique_lab = np.unique(local_labels_with_noise)
        cmap = plt.get_cmap('jet', unique_lab.size)
        cmap = { k: cmap(l) for l, k in enumerate(unique_lab) }
        cmap[-1] = 'k'
        active_ind = np.arange(local_feature.shape[0])
        for k in unique_lab:
            plot_mask_1 = (active_ind < valid_size) & (local_labels_with_noise == k)
            plot_mask_2 = (active_ind >= valid_size) & (local_labels_with_noise == k)
            ax.scatter(local_feature_plot[plot_mask_1, 0], local_feature_plot[plot_mask_1, 1], color=cmap[k], marker='o', alpha=0.3, s=1)
            ax.scatter(local_feature_plot[plot_mask_2, 0], local_feature_plot[plot_mask_2, 1], color=cmap[k], marker='*', alpha=0.3, s=1)
            
        #~ plt.show()

    
    return local_labels_with_noise
    

def _split_waveforms_nested(wfs_and_noise, noise_size, nbefore, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug):

    import sklearn.decomposition
    import hdbscan
    
    valid_size = wfs_and_noise.shape[0] - noise_size
    
    local_labels_with_noise =  np.zeros(wfs_and_noise.shape[0], dtype=np.int64)
    
    local_count = 1
    while True:
        #~ print('  local_count', local_count, np.sum(local_labels_with_noise[:-noise_size] == 0), local_labels_with_noise.size - noise_size)
        
        if np.all(local_labels_with_noise[:-noise_size] != 0):
            break
        
        active_ind, = np.nonzero(local_labels_with_noise == 0)
        
        # reduce dimention in 2 step
        active_wfs = wfs_and_noise[active_ind, :, :]
        local_feature = np.zeros((active_wfs.shape[0], n_components_by_channel * active_wfs.shape[2]))
        tsvd = sklearn.decomposition.TruncatedSVD(n_components=n_components_by_channel)
        for c in range(wfs_and_noise.shape[2]):
            local_feature[:, c*n_components_by_channel:(c+1)*n_components_by_channel] = tsvd.fit_transform(active_wfs[:, :, c])
        #~ n_components = min(n_components, local_feature.shape[1])
        #~ pca = sklearn.decomposition.PCA(n_components=n_components, whiten=True)
        #~ local_feature = pca.fit_transform(local_feature)
        
        # hdbscan on pca
        clustering = hdbscan.hdbscan(local_feature, **hdbscan_params)
        active_labels_with_noise = clustering[0]
        cluster_probability = clustering[2]
        persistent_clusters, = np.nonzero(clustering[2] > probability_thr)
        active_labels_with_noise[~np.in1d(active_labels_with_noise, persistent_clusters)] = -1
        
        active_labels = active_labels_with_noise[active_ind < valid_size]
        active_labels_set = np.unique(active_labels)
        active_labels_set = active_labels_set[active_labels_set>=0]
        num_cluster = active_labels_set.size

        if debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            import umap

            reducer = umap.UMAP()
            local_feature_plot = reducer.fit_transform(local_feature)

            
            unique_lab = np.unique(active_labels_with_noise)
            cmap = plt.get_cmap('jet', unique_lab.size)
            cmap = { k: cmap(l) for l, k in enumerate(unique_lab) }
            cmap[-1] = 'k'
            cmap[-2] = 'b'
            for k in unique_lab:
                plot_mask_1 = (active_ind < valid_size) & (active_labels_with_noise == k)
                plot_mask_2 = (active_ind >= valid_size) & (active_labels_with_noise == k)
                ax.scatter(local_feature_plot[plot_mask_1, 0], local_feature_plot[plot_mask_1, 1], color=cmap[k], marker='o')
                ax.scatter(local_feature_plot[plot_mask_2, 0], local_feature_plot[plot_mask_2, 1], color=cmap[k], marker='*')
                
            #~ plt.show()

        if num_cluster > 1:
            # take the best one
            extremum_values = []
            assert active_ind.size ==  active_labels_with_noise.size
            for k in active_labels_set:
                #~ sel = active_labels_with_noise == k
                #~ sel[-noise_size:] = False
                sel = (active_ind < valid_size) & (active_labels_with_noise == k)
                if np.sum(sel) == 1:
                    # only one spike
                    extremum_values.append(0)
                else:
                    v = np.mean(np.abs(np.mean(active_wfs[sel, nbefore, :], axis=0)))
                    extremum_values.append(v)
            best_label = active_labels_set[np.argmax(extremum_values)]
            #~ inds = active_ind[active_labels_with_noise == best_label]
            inds = active_ind[(active_ind < valid_size) & (active_labels_with_noise == best_label)]
            #~ inds_no_noise = inds[inds < valid_size]
            if inds.size > 1:
                # avoid cluster with one spike
                local_labels_with_noise[inds] = local_count
                local_count += 1
            else:
                local_labels_with_noise[inds] = -1
            
            local_count += 1
            
        elif num_cluster == 1:
            best_label = active_labels_set[0]
            #~ inds = active_ind[active_labels_with_noise == best_label]
            #~ inds_no_noise = inds[inds < valid_size]
            inds = active_ind[(active_ind < valid_size) & (active_labels_with_noise == best_label)]
            if inds.size > 1:
                # avoid cluster with one spike
                local_labels_with_noise[inds] = local_count
                local_count += 1
            else:
                local_labels_with_noise[inds] = -1
                
            # last loop
            local_labels_with_noise[active_ind[active_labels_with_noise==-1]] = -1
        else:
            local_labels_with_noise[active_ind] = -1
            break
    
    #~ local_labels = local_labels_with_noise[:-noise_size]
    local_labels_with_noise[local_labels_with_noise>0] -= 1
    
    return local_labels_with_noise
    


def auto_split_clustering(wfs_arrays, sparsity_mask, labels, peak_labels,  nbefore, nafter, noise,
                n_components_by_channel=3,
                n_components=5,
                hdbscan_params={},
                probability_thr=0,
                debug=False,
                debug_folder=None,
                ):
    """
    Loop over sparse waveform and try to over split.
    Internally used by PositionAndPCAClustering for the second step.
    """
    
    import sklearn.decomposition
    import hdbscan
    
    split_peak_labels = -1 * np.ones(peak_labels.size, dtype=np.int64)
    nb_clusters = 0
    main_channels = {}
    for l, label in enumerate(labels):
        #~ print()
        #~ print('auto_split_clustering', label, l, len(labels))
        
        chans, = np.nonzero(sparsity_mask[l, :])
        
        wfs = wfs_arrays[label]
        valid_size = wfs.shape[0]

        wfs_and_noise = np.concatenate([wfs, noise[:, :, chans]], axis=0)
        noise_size = noise.shape[0]
        
        local_labels_with_noise = _split_waveforms(wfs_and_noise, noise_size, nbefore, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug)
        # local_labels_with_noise = _split_waveforms_nested(wfs_and_noise, noise_size, nbefore, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug)

        local_labels = local_labels_with_noise[:valid_size]
        if noise_size > 0:
            local_labels_with_noise[valid_size:] = -2
        
        
        for k in  np.unique(local_labels):
            if k < 0:
                continue
            template = np.mean(wfs[local_labels == k, :, :], axis=1)
            chan_inds, = np.nonzero(sparsity_mask[l, :])
            assert wfs.shape[2] == chan_inds.size
            main_chan = chan_inds[np.argmax(np.max(np.abs(template), axis=0))]
            main_channels[k + nb_clusters] = main_chan

        if debug:
            #~ local_labels_with_noise[-noise_size:] = -2
            import matplotlib.pyplot as plt
            #~ fig, axs = plt.subplots(ncols=3)
            
            fig, ax = plt.subplots()
            plot_labels_set = np.unique(local_labels_with_noise)
            cmap = plt.get_cmap('jet', plot_labels_set.size)
            cmap = { k: cmap(l) for l, k in enumerate(plot_labels_set) }
            cmap[-1] = 'k'
            cmap[-2] = 'b'
            
            for plot_label in plot_labels_set:
                plot_mask = (local_labels_with_noise == plot_label)
                color = cmap[plot_label]
                
                wfs_flat = wfs_and_noise[plot_mask, :, :].swapaxes(1, 2).reshape(np.sum(plot_mask), -1).T
                ax.plot(wfs_flat, color=color, alpha=0.1)
                if plot_label >=0:
                    ax.plot(wfs_flat.mean(1), color=color, lw=2)
            
            for c in range(wfs.shape[2]):
                ax.axvline(c * (nbefore + nafter) + nbefore, color='k', ls='-', alpha=0.5)
            
            if debug_folder is not None:
                fig.savefig(debug_folder / f'auto_split_{label}.png')
            
            plt.show()
        
        # remove noise labels
        mask, = np.nonzero(peak_labels == label)
        mask2, = np.nonzero(local_labels >= 0)
        split_peak_labels[mask[mask2]] = local_labels[mask2] + nb_clusters

        nb_clusters += local_labels.max() + 1
    
    return split_peak_labels, main_channels
    



def auto_clean_clustering(wfs_arrays, sparsity_mask, labels, peak_labels, nbefore, nafter, channel_distances,
            radius_um=50, auto_merge_num_shift=7, auto_merge_quantile_limit=0.8, ratio_num_channel_intersect=0.8):
    """
    
    
    """
    clean_peak_labels = peak_labels.copy()

    #~ labels = np.unique(peak_labels)
    #~ labels = labels[labels >= 0]
    
    # Check debug
    assert sparsity_mask.shape[0] == labels.shape[0]
    
    # get main channel on new wfs
    templates = []
    main_channels = []
    
    for l, label in enumerate(labels):
        wfs = wfs_arrays[label]
        template = np.mean(wfs, axis=0)
        chan_inds, = np.nonzero(sparsity_mask[l, :])
        assert wfs.shape[2] == chan_inds.size
        main_chan = chan_inds[np.argmax(np.max(np.abs(template), axis=0))]
        templates.append(template)
        main_channels.append(main_chan)
        
        #~ plot_debug = True
        #~ if plot_debug:
            #~ import matplotlib.pyplot as plt
            #~ wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T
            #~ fig, ax = plt.subplots()
            #~ ax.plot(wfs_flat, color='g', alpha=0.1)
            #~ ax.plot(template.T.flatten(), color='c', lw=2)
            #~ plt.show()

    ## Step 1 : auto merge
    auto_merge_list = []
    for l0 in  range(len(labels)):
        for l1 in  range(l0+1, len(labels)):
            
            label0, label1 = labels[l0], labels[l1]

            main_chan0 = main_channels[l0]
            main_chan1 = main_channels[l1]
            
            if channel_distances[main_chan0, main_chan1] > radius_um:
                continue
            
            
            channel_inds0, = np.nonzero(sparsity_mask[l0, :])
            channel_inds1, = np.nonzero(sparsity_mask[l1, :])
            
            intersect_chans = np.intersect1d(channel_inds0, channel_inds1)
            # union_chans = np.union1d(channel_inds0, channel_inds1)
            
            # we use
            radius_chans, = np.nonzero((channel_distances[main_chan0, :] <= radius_um) | (channel_distances[main_chan1, :] <= radius_um))
            if radius_chans.size < (intersect_chans.size * ratio_num_channel_intersect):
                #~ print('WARNING INTERSECT')
                #~ print(intersect_chans.size, radius_chans.size, used_chans.size)
                continue

            used_chans = np.intersect1d(radius_chans, intersect_chans)
            
            if used_chans.size == 0:
                continue
                
            
            wfs0 = wfs_arrays[label0]
            wfs0 = wfs0[:, :,np.in1d(channel_inds0, used_chans)]
            wfs1 = wfs_arrays[label1]
            wfs1 = wfs1[:, :,np.in1d(channel_inds1, used_chans)]
            
            # TODO : remove
            assert wfs0.shape[2] == wfs1.shape[2]
            
            equal, shift = check_equal_template_with_distribution_overlap(wfs0, wfs1,
                            num_shift=auto_merge_num_shift, quantile_limit=auto_merge_quantile_limit, 
                            return_shift=True, debug=False)
            
            if equal:
                auto_merge_list.append((l0, l1, shift))

            # DEBUG plot
            #~ plot_debug = debug
            #~ plot_debug = True
            plot_debug = False
            #~ plot_debug = equal
            if plot_debug :
                import matplotlib.pyplot as plt
                wfs_flat0 = wfs0.swapaxes(1, 2).reshape(wfs0.shape[0], -1).T
                wfs_flat1 = wfs1.swapaxes(1, 2).reshape(wfs1.shape[0], -1).T
                fig, ax = plt.subplots()
                ax.plot(wfs_flat0, color='g', alpha=0.1)
                ax.plot(wfs_flat1, color='r', alpha=0.1)

                ax.plot(np.mean(wfs_flat0, axis=1), color='c', lw=2)
                ax.plot(np.mean(wfs_flat1, axis=1), color='m', lw=2)
                
                for c in range(len(used_chans)):
                    ax.axvline(c * (nbefore + nafter) + nbefore, color='k', ls='--')
                ax.set_title(f'label0={label0} label1={label1} equal{equal} shift{shift} \n chans intersect{intersect_chans.size} radius{radius_chans.size} used{used_chans.size}')
                plt.show()
    
    #~ print('auto_merge_list', auto_merge_list)
    # merge in reverse order because of shift accumulation
    peak_sample_shifts = np.zeros(peak_labels.size, dtype='int64')
    for (l0, l1, shift) in auto_merge_list[::-1]:
        label0, label1 = labels[l0], labels[l1]
        inds, = np.nonzero(peak_labels == label1)
        clean_peak_labels[inds] = label0
        peak_sample_shifts[inds] += shift

    # update label list
    labels_clean = np.unique(clean_peak_labels)
    labels_clean = labels_clean[labels_clean >= 0]
    
    # Step 2 : remove none aligner units
    # some unit have a secondary peak that can be detected
    # lets remove the secondary template
    # to avoid recomputing template this is done on the original list
    auto_trash_list = []
    auto_trash_misalignment_shift = auto_merge_num_shift + 1
    for l, label in enumerate(labels):
        if label not in labels_clean:
            continue
        
        template = templates[l]
        main_chan = main_channels[l]
        chan_inds,  = np.nonzero(sparsity_mask[l, :])
        max_ind = list(chan_inds).index(main_chan)
        sample_max = np.argmax(np.abs(template[:, max_ind]))
        
        not_aligned = np.abs(sample_max - nbefore) >= auto_trash_misalignment_shift
        if not_aligned:
            auto_trash_list.append(label)

        #~ plot_debug = not_aligned
        #~ plot_debug = True
        plot_debug = False
        #~ plot_debug = label in (23, )
        if plot_debug :
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(nrows=2)
            ax = axs[0]
            wfs_flat = template.T.flatten()
            ax.plot(wfs_flat)
            for c in range(template.shape[1]):
                ax.axvline(c * (nbefore + nafter) + nbefore, color='k', ls='--')
                if c == max_ind:
                    ax.axvline(c * (nbefore + nafter) + sample_max, color='b')
                    
            ax.set_title(f'label={label}  not_aligned{not_aligned} chan_inds={chan_inds} chan={chan_inds[max_ind]} max_ind{max_ind}')
            ax = axs[1]
            ax.plot(template[:, max_ind])
            ax.axvline(sample_max)
            ax.axvline(nbefore, color='r', ls='--')
            plt.show()
    
    #~ print('auto_trash_list', auto_trash_list)
    for label in auto_trash_list:
        inds, = np.nonzero(clean_peak_labels == label)
        clean_peak_labels[inds] = -label


    return clean_peak_labels, peak_sample_shifts
    
    
    
    

# %%
def run_LDA_max_chan_split(wfs, max_channels, threshold_diptest=.5):
    ncomp = 2
    if np.unique(max_channels).shape[0] < 2:
        return np.zeros(len(max_channels), dtype=int)
    elif np.unique(max_channels).shape[0] == 2:
        ncomp = 1
    try:
        lda_model = LDA(n_components=ncomp)
        lda_comps = lda_model.fit_transform(
            wfs.reshape((-1, wfs.shape[1] * wfs.shape[2])), max_channels
        )
    except np.linalg.LinAlgError:
        nmc = np.unique(max_channels).shape[0]
        print("SVD error, skipping this one. N maxchans was", nmc, "n data", len(wfs))
        return np.zeros(len(max_channels), dtype=int)
    except ValueError as e:
        print("Some ValueError during LDA split. Ignoring it and not splitting.")
        print("Here is the error message though", e, flush=True)
        return np.zeros(len(max_channels), dtype=int)
    
    if ncomp == 2:
        lda_clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
        lda_clusterer.fit(lda_comps)
        labels = lda_clusterer.labels_
    else:
        value_dpt, cut_value = isocut5(lda_comps[:,0])
        if value_dpt < threshold_diptest:
            labels = np.zeros(len(max_channels), dtype=int)
        else:
            labels = np.zeros(len(max_channels), dtype=int)
            labels[np.where(lda_comps[:,0] > cut_value)] = 1   
            
    return labels


def split_individual_cluster(
    true_mc,
    waveforms_unit,
    x_unit,
    z_unit, 
    geom_array,
    denoiser,
    device,
    tpca, #WHERE TO PUT TPCA TRAINING
    pca_n_channels,
    threshold_diptest,
    min_size_split=25,
):
    total_channels = geom_array.shape[0]
    N, T, wf_chans = waveforms_unit.shape
    n_channels_half = pca_n_channels//2

    labels_unit = np.full(waveforms_unit.shape[0], -1)
    is_split = False
    
    if N < min_size_split:
        return is_split, labels_unit * 0

    mc = max(n_channels_half, true_mc)
    mc = min(total_channels - n_channels_half, mc)
    assert mc - n_channels_half >= 0
    assert mc + n_channels_half <= total_channels

    wfs_unit = waveforms_unit[:, :, mc-n_channels_half:mc+n_channels_half]
    #DO WE NEED TO TRANSPOSE AFTER THIS? SHOULD BE N,C,T after transpose 
    permuted_wfs_unit = wfs_unit.transpose(0, 2, 1)

    #get tpca of wfs using pre-trained tpca
    tpca_wf_units = tpca.transform(permuted_wfs_unit.reshape(permuted_wfs_unit.shape[0]*permuted_wfs_unit.shape[1], -1))
    tpca_wfs_inverse = tpca.inverse_transform(tpca_wf_units)
    tpca_wfs_inverse = tpca_wfs_inverse.reshape(permuted_wfs_unit.shape[0], permuted_wfs_unit.shape[1], -1).transpose(0, 2, 1)
    tpca_wf_units = tpca_wf_units.reshape(permuted_wfs_unit.shape[0], permuted_wfs_unit.shape[1], -1).transpose(0, 2, 1)
    
    ptps_unit = wfs_unit[:, :, n_channels_half].ptp(1)
    
    #get tpca embeddings for pca_n_channels (edges handled differently)
    tpca_wf_units = tpca_wf_units.transpose(0, 2, 1)
    tpca_wf_units = tpca_wf_units.reshape(tpca_wf_units.shape[0], tpca_wf_units.shape[1]*tpca_wf_units.shape[2])
    
    #get 2D pc embedding of tpca embeddings
    pca_model = PCA(2)
    try:
        pcs = pca_model.fit_transform(tpca_wf_units)
    except ValueError:
        print("ERR", tpca_wf_units.shape, flush=True)
        raise
    
    #scale pc embeddings to X feature
    alpha1 = (x_unit.max() - x_unit.min()) / (
        pcs[:, 0].max() - pcs[:, 0].min()
    )
    alpha2 = (x_unit.max() - x_unit.min()) / (
        pcs[:, 1].max() - pcs[:, 1].min()
    )
    
    #create 5D feature set for clustering (herdingspikes)
    features = np.concatenate(
        (
            np.expand_dims(x_unit, 1),
            np.expand_dims(z_unit, 1),
            np.expand_dims(pcs[:, 0], 1) * alpha1,
            np.expand_dims(pcs[:, 1], 1) * alpha2,
            np.expand_dims(np.log(ptps_unit) * 30, 1),
        ),
        axis=1,
    )  # Use scales parameter
    
    #cluster using herding spikes (parameters could be adjusted)
    clusterer_herding = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    clusterer_herding.fit(features)
    labels_rec_hdbscan = clusterer_herding.labels_
    #check if cluster split by herdingspikes clustering
    if np.unique(labels_rec_hdbscan).shape[0] > 1:
        is_split = True
        
    #LDA split - split by clustering LDA embeddings: X,y = wfs,max_channels
    max_channels_all = wfs_unit.ptp(1).argmax(1)
    if is_split:
        #split by herdingspikes, run LDA split on new clusters.
        labels_unit[labels_rec_hdbscan == -1] = -1
        label_max_temp = labels_rec_hdbscan.max()
        cmp = 0
        for new_unit_id in np.unique(labels_rec_hdbscan)[1:]:
            tpca_wfs_new_unit = tpca_wf_units[labels_rec_hdbscan == new_unit_id]
            #get max_channels for new unit
            max_channels = wfs_unit[labels_rec_hdbscan == new_unit_id].ptp(1).argmax(1)
            #lda split
            lda_labels = run_LDA_max_chan_split(tpca_wfs_new_unit, max_channels, threshold_diptest)
            if np.unique(lda_labels).shape[0] == 1:
                labels_unit[labels_rec_hdbscan == new_unit_id] = cmp
                cmp += 1
            else:
                for lda_unit in np.unique(lda_labels):
                    if lda_unit >= 0:
                        labels_unit[
                            np.flatnonzero(labels_rec_hdbscan == new_unit_id)[
                                lda_labels == lda_unit
                            ]
                        ] = cmp
                        cmp += 1
                    else:
                        labels_unit[
                            np.flatnonzero(labels_rec_hdbscan == new_unit_id)[
                                lda_labels == lda_unit
                            ]
                        ] = -1
    else:
        #not split by herdingspikes, run LDA split.
        lda_labels = run_LDA_max_chan_split(tpca_wf_units, max_channels_all, threshold_diptest)
        if np.unique(lda_labels).shape[0] > 1:
            is_split = True
            labels_unit = lda_labels
    # print("split", is_split, np.unique(labels_unit), len(np.unique(labels_unit)[1:]))
    return is_split, labels_unit


# %%
def split_clusters(
    waveforms, #already aligned to template
    template_maxchans,
    labels,
    x,
    z, #Y is the probe plane in SI
    geom_array,
    denoiser,
    device,
    tpca, #not sure? What should we do here? Perform tpca before?
    n_channels=10,
    pca_n_channels=4,
    threshold_diptest=.5,
):
    labels_new = labels.copy()
    labels_original = labels.copy()
    cur_max_label = labels.max()
    for unit in (np.setdiff1d(np.unique(labels), [-1])): #216
        unit = int(unit)
        # print(f"splitting unit {unit}")
        in_unit = np.flatnonzero(labels == unit)
        waveforms_unit = waveforms[in_unit]  
        x_unit, z_unit = x[in_unit], z[in_unit]

        is_split, unit_new_labels = split_individual_cluster(
            template_maxchans[unit],
            waveforms_unit,
            x_unit,
            z_unit,
            geom_array,
            denoiser,
            device,
            tpca,
            n_channels,
            pca_n_channels,
            threshold_diptest,
        )
        if is_split:
            for new_label in np.unique(unit_new_labels):
                if new_label == -1:
                    idx = np.flatnonzero(labels_original == unit)[
                        unit_new_labels == new_label
                    ]
                    labels_new[idx] = new_label
                elif new_label > 0:
                    cur_max_label += 1
                    idx = np.flatnonzero(labels_original == unit)[
                        unit_new_labels == new_label
                    ]
                    labels_new[idx] = cur_max_label
    return labels_new


# %%
def get_x_z_templates(n_templates, labels, x, z):
    x_z_templates = np.zeros((n_templates, 2))
    for i in range(n_templates):
        x_z_templates[i, 1] = np.median(z[labels == i])
        x_z_templates[i, 0] = np.median(x[labels == i])
    return x_z_templates


# %%
def get_n_spikes_templates(n_templates, labels):
    n_spikes_templates = np.zeros(n_templates)
    for i in range(n_templates):
        n_spikes_templates[i] = (labels == i).sum()
    return n_spikes_templates



# %%
def get_proposed_pairs(
    n_templates, templates, x_z_templates, n_temp=20, n_channels=10
):
    n_channels_half = n_channels // 2
    dist = cdist(x_z_templates, x_z_templates)
    dist_argsort = dist.argsort(axis=1)[:, 1 : n_temp + 1]
    dist_template = np.zeros((dist_argsort.shape[0], n_temp))
    for i in range(n_templates):
        mc = min(templates[i].ptp(0).argmax(), 384 - n_channels_half)
        mc = max(mc, n_channels_half)
        temp_a = templates[i, :, mc - n_channels_half : mc + n_channels_half]
        for j in range(n_temp):
            temp_b = templates[
                dist_argsort[i, j],
                :,
                mc - n_channels_half : mc + n_channels_half,
            ]
            dist_template[i, j] = compute_shifted_similarity(temp_a, temp_b)[0]
    return dist_argsort, dist_template


def compute_shifted_similarity(template1, template2, shifts=[0]):
    curr_similarities = []
    for shift in shifts:
        if shift == 0:
            similarity = np.max(np.abs(template1 - template2))
        elif shift < 0:
            template2_shifted_flattened = np.pad(template2.T.flatten(),((-shift,0)), mode='constant')[:shift]
            similarity = np.max(np.abs(template1.T.flatten() - template2_shifted_flattened))
        else:    
            template2_shifted_flattened = np.pad(template2.T.flatten(),((0,shift)), mode='constant')[shift:]
            similarity = np.max(np.abs(template1.T.flatten() - template2_shifted_flattened))
        curr_similarities.append(similarity)
    return np.min(curr_similarities), shifts[np.argmin(curr_similarities)]


# %%
def get_diptest_value(
    waveforms, #already aligned
    geom_array,
    labels,
    unit_a,
    unit_b,
    n_spikes_templates,
    mc,
    two_units_shift,
    unit_shifted,
    denoiser,
    device,
    tpca,
    n_channels=10,
    n_times=121,
    max_spikes=250,
    nn_denoise = True
):
    # ALIGN BASED ON MAX PTP TEMPLATE MC
    n_channels_half = n_channels // 2

    n_wfs_max = int(
        min(max_spikes, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))
    )

    mc = min(384 - n_channels_half, mc)
    mc = max(n_channels_half, mc)

    idx = np.random.choice(
        np.arange((labels == unit_a).sum()), n_wfs_max, replace=False
    )
    # print(spike_times_unit_a.shape)
    idx.sort()
    wfs_a = waveforms[labels == unit_a][idx]

    idx = np.random.choice(
        np.arange((labels == unit_b).sum()), n_wfs_max, replace=False
    )
    idx.sort()
    wfs_b = waveforms[labels == unit_b][idx]

    wfs_a_bis = np.zeros((wfs_a.shape[0], n_times, n_channels))
    wfs_b_bis = np.zeros((wfs_b.shape[0], n_times, n_channels))

    if two_units_shift > 0:

        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                wfs_a_bis[i, :-two_units_shift] = wfs_a[
                    i, two_units_shift:, mc - n_channels_half: mc + n_channels_half
                ]
                wfs_b_bis[i, :] = wfs_b[
                    i, :, mc - n_channels_half: mc + n_channels_half
                ]
        else:
            for i in range(wfs_a_bis.shape[0]):
                wfs_a_bis[i] = wfs_a[
                    i, :, mc - n_channels_half: mc + n_channels_half
                ]
                wfs_b_bis[i, :-two_units_shift] = wfs_b[
                    i, two_units_shift:, mc - n_channels_half: mc + n_channels_half
                ]
    elif two_units_shift < 0:
        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                wfs_a_bis[i, -two_units_shift:] = wfs_a[
                    i, :two_units_shift, mc - n_channels_half: mc + n_channels_half
                ]
                wfs_b_bis[i, :] = wfs_b[
                    i, :, mc - n_channels_half: mc + n_channels_half
                ]
        else:
            for i in range(wfs_a_bis.shape[0]):
                wfs_a_bis[i] = wfs_a[
                    i, :, mc - n_channels_half: mc + n_channels_half
                ]
                wfs_b_bis[i, -two_units_shift:] = wfs_b[
                    i, :two_units_shift, mc - n_channels_half: mc + n_channels_half
                ]
    else:
        for i in range(wfs_a_bis.shape[0]):
            wfs_a_bis[i] = wfs_a[i, :, mc - n_channels_half: mc + n_channels_half]
            wfs_b_bis[i, :] = wfs_b[i, :, mc - n_channels_half: mc + n_channels_half]

    # tpca = PCA(rank_pca)
    wfs_diptest = np.concatenate((wfs_a_bis, wfs_b_bis))

    if nn_denoise:
        #Import from spikeinterface
        wfs_diptest = denoise_wf_nn_tmp_single_channel(
            wfs_diptest, denoiser, device
        )
    # print(wfs_diptest.shape)
    N, T, C = wfs_diptest.shape
    wfs_diptest = wfs_diptest.transpose(0, 2, 1).reshape(N * C, T)

    wfs_diptest = tpca.fit_transform(wfs_diptest)
    wfs_diptest = (
        wfs_diptest.reshape(N, C, tpca.n_components).transpose(0, 2, 1).reshape((N, C * tpca.n_components))
    )
    labels_diptest = np.zeros(wfs_a_bis.shape[0] + wfs_b_bis.shape[0])
    labels_diptest[: wfs_a_bis.shape[0]] = 1

    lda_model = LDA(n_components=1)
    lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
    value_dpt, cut_calue = isocut5(lda_comps[:, 0]) #Charlie pushing this -> Import correctly after PR
    return value_dpt


# %%
def merge_clusters(
    waveforms,
    geom_array,
    templates,
    n_templates,
    labels,
    x,
    z, #Y HERE?
    denoiser,
    device,
    tpca, #TRAIN BEFORE
    n_channels=10,
    n_temp=10,
    distance_threshold=3.0,
    threshold_diptest=0.75,
    nn_denoise=False,
):

    n_spikes_templates = get_n_spikes_templates(n_templates, labels)
    x_z_templates = get_x_z_templates(n_templates, labels, x, z)
    print("GET PROPOSED PAIRS")

    dist_argsort, dist_template = get_proposed_pairs( #get_merge_candidates
        n_templates, templates, x_z_templates, n_temp=n_temp
    )

    labels_updated = labels.copy()
    reference_units = np.setdiff1d(np.unique(labels), [-1])

    for unit in (range(n_templates)):
        unit_reference = reference_units[unit]
        to_be_merged = [unit_reference]
        merge_shifts = [0]
        is_merged = False

        for j in range(n_temp):
            if dist_template[unit, j] < distance_threshold:
                unit_bis = dist_argsort[unit, j]
                unit_bis_reference = reference_units[unit_bis]
                if unit_reference != unit_bis_reference:
                    # ALIGN BASED ON MAX PTP TEMPLATE MC
                    if (
                        templates[unit_reference].ptp(0).max()
                        < templates[unit_bis_reference].ptp(0).max()
                    ):
                        mc = templates[unit_bis_reference].ptp(0).argmax()
                        two_units_shift = (
                            templates[unit_reference, :, mc].argmin()
                            - templates[unit_bis_reference, :, mc].argmin()
                        )
                        unit_shifted = unit_reference
                    else:
                        mc = templates[unit_reference].ptp(0).argmax()
                        two_units_shift = (
                            templates[unit_bis_reference, :, mc].argmin()
                            - templates[unit_reference, :, mc].argmin()
                        )
                        unit_shifted = unit_bis_reference

                    dpt_val = get_diptest_value(
                        waveforms,
                        first_chans,
                        geom_array,
                        labels_updated,
                        unit_reference,
                        unit_bis_reference,
                        n_spikes_templates,
                        mc,
                        two_units_shift,
                        unit_shifted,
                        denoiser,
                        device,
                        tpca,
                        n_channels,
                        nn_denoise=nn_denoise,
                    )
                    if (
                        dpt_val < threshold_diptest
                        and np.abs(two_units_shift) < 2
                    ):
                        to_be_merged.append(unit_bis_reference)
                        if unit_shifted == unit_bis_reference:
                            merge_shifts.append(-two_units_shift)
                        else:
                            merge_shifts.append(two_units_shift)
                        is_merged = True
        if is_merged:
            n_total_spikes = 0
            for unit_merged in np.unique(np.asarray(to_be_merged)):
                n_total_spikes += n_spikes_templates[unit_merged]

            new_reference_unit = to_be_merged[0]

            templates[new_reference_unit] = (
                n_spikes_templates[new_reference_unit]
                * templates[new_reference_unit]
                / n_total_spikes
            )
            cmp = 1
            for unit_merged in to_be_merged[1:]:
                shift_ = merge_shifts[cmp]
                templates[new_reference_unit] += (
                    n_spikes_templates[unit_merged]
                    * np.roll(templates[unit_merged], shift_, axis=0)
                    / n_total_spikes
                )
                n_spikes_templates[new_reference_unit] += n_spikes_templates[
                    unit_merged
                ]
                n_spikes_templates[unit_merged] = 0
                labels_updated[
                    labels_updated == unit_merged
                ] = new_reference_unit
                reference_units[reference_units == unit_merged] = new_reference_unit
                cmp += 1
    return labels_updated
