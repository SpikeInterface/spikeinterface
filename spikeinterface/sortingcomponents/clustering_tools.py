"""
This gather some function usefull for some clusetring algos.
"""

import numpy as np

from ..toolkit import check_equal_template_with_distribution_overlap


#~ def concatenate_sparse_waveforms(wfs_arrays, unit_ids, select_unit_ids, select_channel_inds, sparsity_mask, do_concatenate=True):
    #~ """
    #~ Helper function that concatenate several waveform buffer from different unit_ids
    #~ that have diffrent channel shape because of sparsity and concatenate then.
    #~ """


    #~ unit_ids = list(unit_ids)
    #~ select_unit_ids = [unit_ids.index(unit_id) for unit_id in select_unit_ids]

    #~ assert np.all(sparsity_mask[select_unit_ids, select_channel_inds], axis=0), 'Same channel are sparse for some units'

    #~ all_wfs = []

    #~ for unit_id in select_unit_ids:
        #~ wfs = wfs_arrays[unit_id]
        #~ unit_index = unit_ids.index(unit_id)
        #~ chans, = np.nonzero(sparsity_mask[unit_index, :])
        #~ wfs_chan = wfs[:, :, np.in1d(chans, select_channel_inds)]
        #~ all_wfs.append(wfs_chan)
    
    #~ if do_concatenate:
        #~ all_wfs = np.concatenate(all_wfs, axis=0)

    #~ return all_wfs



def auto_clean_clustering(wfs_arrays, sparsity_mask, labels, peak_labels, nbefore, nafter, channel_distances,
            radius_um=50, auto_merge_num_shift=7, auto_merge_quantile_limit=0.8):
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
            # union_chans = np.union1d(channel_inds0, channel_inds1)
            
            # we use
            radius_chans, = np.nonzero((channel_distances[main_chan0, :] <= radius_um) & (channel_distances[main_chan1, :] <= radius_um))
            used_chans = np.intersect1d(radius_chans, intersect_chans)
            
            if len(used_chans) == 0:
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
                ax.set_title(f'label0={label0} label1={label1} equal{equal} shift{shift}')
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
    
    