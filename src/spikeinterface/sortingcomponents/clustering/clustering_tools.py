"""
This gather some function usefull for some clusetring algos.
"""

from __future__ import annotations


import numpy as np
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.postprocessing import check_equal_template_with_distribution_overlap
from spikeinterface.core import NumpySorting


def _split_waveforms(
    wfs_and_noise, noise_size, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug
):
    import sklearn.decomposition
    import hdbscan

    valid_size = wfs_and_noise.shape[0] - noise_size

    local_feature = np.zeros((wfs_and_noise.shape[0], n_components_by_channel * wfs_and_noise.shape[2]))
    tsvd = sklearn.decomposition.TruncatedSVD(n_components=n_components_by_channel)
    for c in range(wfs_and_noise.shape[2]):
        local_feature[:, c * n_components_by_channel : (c + 1) * n_components_by_channel] = tsvd.fit_transform(
            wfs_and_noise[:, :, c]
        )
    n_components = min(n_components, local_feature.shape[1])
    pca = sklearn.decomposition.PCA(n_components=n_components, whiten=True)
    local_feature = pca.fit_transform(local_feature)

    # hdbscan on pca
    clustering = hdbscan.hdbscan(local_feature, **hdbscan_params)
    local_labels_with_noise = clustering[0]
    cluster_probability = clustering[2]
    (persistent_clusters,) = np.nonzero(cluster_probability > probability_thr)
    local_labels_with_noise[~np.isin(local_labels_with_noise, persistent_clusters)] = -1

    # remove super small cluster
    labels, count = np.unique(local_labels_with_noise[:valid_size], return_counts=True)
    mask = labels >= 0
    labels, count = labels[mask], count[mask]
    minimum_cluster_size_ratio = 0.05

    # ~ print(labels, count)

    to_remove = labels[(count / valid_size) < minimum_cluster_size_ratio]
    # ~ print('to_remove', to_remove, count / valid_size)
    if to_remove.size > 0:
        local_labels_with_noise[np.isin(local_labels_with_noise, to_remove)] = -1

    local_labels_with_noise[valid_size:] = -2

    if debug:
        import matplotlib.pyplot as plt
        import umap

        fig, ax = plt.subplots()

        # ~ reducer = umap.UMAP()
        # ~ local_feature_plot = reducer.fit_transform(local_feature)
        local_feature_plot = local_feature

        unique_lab = np.unique(local_labels_with_noise)
        cmap = plt.colormaps["jet"].resampled(unique_lab.size)
        cmap = {k: cmap(l) for l, k in enumerate(unique_lab)}
        cmap[-1] = "k"
        active_ind = np.arange(local_feature.shape[0])
        for k in unique_lab:
            plot_mask_1 = (active_ind < valid_size) & (local_labels_with_noise == k)
            plot_mask_2 = (active_ind >= valid_size) & (local_labels_with_noise == k)
            ax.scatter(
                local_feature_plot[plot_mask_1, 0],
                local_feature_plot[plot_mask_1, 1],
                color=cmap[k],
                marker="o",
                alpha=0.3,
                s=1,
            )
            ax.scatter(
                local_feature_plot[plot_mask_2, 0],
                local_feature_plot[plot_mask_2, 1],
                color=cmap[k],
                marker="*",
                alpha=0.3,
                s=1,
            )

        # ~ plt.show()

    return local_labels_with_noise


def _split_waveforms_nested(
    wfs_and_noise, noise_size, nbefore, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug
):
    import sklearn.decomposition
    import hdbscan

    valid_size = wfs_and_noise.shape[0] - noise_size

    local_labels_with_noise = np.zeros(wfs_and_noise.shape[0], dtype=np.int64)

    local_count = 1
    while True:
        # ~ print('  local_count', local_count, np.sum(local_labels_with_noise[:-noise_size] == 0), local_labels_with_noise.size - noise_size)

        if np.all(local_labels_with_noise[:-noise_size] != 0):
            break

        (active_ind,) = np.nonzero(local_labels_with_noise == 0)

        # reduce dimention in 2 step
        active_wfs = wfs_and_noise[active_ind, :, :]
        local_feature = np.zeros((active_wfs.shape[0], n_components_by_channel * active_wfs.shape[2]))
        tsvd = sklearn.decomposition.TruncatedSVD(n_components=n_components_by_channel)
        for c in range(wfs_and_noise.shape[2]):
            local_feature[:, c * n_components_by_channel : (c + 1) * n_components_by_channel] = tsvd.fit_transform(
                active_wfs[:, :, c]
            )
        # ~ n_components = min(n_components, local_feature.shape[1])
        # ~ pca = sklearn.decomposition.PCA(n_components=n_components, whiten=True)
        # ~ local_feature = pca.fit_transform(local_feature)

        # hdbscan on pca
        clustering = hdbscan.hdbscan(local_feature, **hdbscan_params)
        active_labels_with_noise = clustering[0]
        cluster_probability = clustering[2]
        (persistent_clusters,) = np.nonzero(clustering[2] > probability_thr)
        active_labels_with_noise[~np.isin(active_labels_with_noise, persistent_clusters)] = -1

        active_labels = active_labels_with_noise[active_ind < valid_size]
        active_labels_set = np.unique(active_labels)
        active_labels_set = active_labels_set[active_labels_set >= 0]
        num_cluster = active_labels_set.size

        if debug:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

            import umap

            reducer = umap.UMAP()
            local_feature_plot = reducer.fit_transform(local_feature)

            unique_lab = np.unique(active_labels_with_noise)
            cmap = plt.colormaps["jet"].resampled(unique_lab.size)
            cmap = {k: cmap(l) for l, k in enumerate(unique_lab)}
            cmap[-1] = "k"
            cmap[-2] = "b"
            for k in unique_lab:
                plot_mask_1 = (active_ind < valid_size) & (active_labels_with_noise == k)
                plot_mask_2 = (active_ind >= valid_size) & (active_labels_with_noise == k)
                ax.scatter(
                    local_feature_plot[plot_mask_1, 0], local_feature_plot[plot_mask_1, 1], color=cmap[k], marker="o"
                )
                ax.scatter(
                    local_feature_plot[plot_mask_2, 0], local_feature_plot[plot_mask_2, 1], color=cmap[k], marker="*"
                )

            # ~ plt.show()

        if num_cluster > 1:
            # take the best one
            extremum_values = []
            assert active_ind.size == active_labels_with_noise.size
            for k in active_labels_set:
                # ~ sel = active_labels_with_noise == k
                # ~ sel[-noise_size:] = False
                sel = (active_ind < valid_size) & (active_labels_with_noise == k)
                if np.sum(sel) == 1:
                    # only one spike
                    extremum_values.append(0)
                else:
                    v = np.mean(np.abs(np.mean(active_wfs[sel, nbefore, :], axis=0)))
                    extremum_values.append(v)
            best_label = active_labels_set[np.argmax(extremum_values)]
            # ~ inds = active_ind[active_labels_with_noise == best_label]
            inds = active_ind[(active_ind < valid_size) & (active_labels_with_noise == best_label)]
            # ~ inds_no_noise = inds[inds < valid_size]
            if inds.size > 1:
                # avoid cluster with one spike
                local_labels_with_noise[inds] = local_count
                local_count += 1
            else:
                local_labels_with_noise[inds] = -1

            local_count += 1

        elif num_cluster == 1:
            best_label = active_labels_set[0]
            # ~ inds = active_ind[active_labels_with_noise == best_label]
            # ~ inds_no_noise = inds[inds < valid_size]
            inds = active_ind[(active_ind < valid_size) & (active_labels_with_noise == best_label)]
            if inds.size > 1:
                # avoid cluster with one spike
                local_labels_with_noise[inds] = local_count
                local_count += 1
            else:
                local_labels_with_noise[inds] = -1

            # last loop
            local_labels_with_noise[active_ind[active_labels_with_noise == -1]] = -1
        else:
            local_labels_with_noise[active_ind] = -1
            break

    # ~ local_labels = local_labels_with_noise[:-noise_size]
    local_labels_with_noise[local_labels_with_noise > 0] -= 1

    return local_labels_with_noise


def auto_split_clustering(
    wfs_arrays,
    sparsity_mask,
    labels,
    peak_labels,
    nbefore,
    nafter,
    noise,
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
        # ~ print()
        # ~ print('auto_split_clustering', label, l, len(labels))

        (chans,) = np.nonzero(sparsity_mask[l, :])

        wfs = wfs_arrays[label]
        valid_size = wfs.shape[0]

        wfs_and_noise = np.concatenate([wfs, noise[:, :, chans]], axis=0)
        noise_size = noise.shape[0]

        local_labels_with_noise = _split_waveforms(
            wfs_and_noise, noise_size, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug
        )
        # local_labels_with_noise = _split_waveforms_nested(wfs_and_noise, noise_size, nbefore, n_components_by_channel, n_components, hdbscan_params, probability_thr, debug)

        local_labels = local_labels_with_noise[:valid_size]
        if noise_size > 0:
            local_labels_with_noise[valid_size:] = -2

        for k in np.unique(local_labels):
            if k < 0:
                continue
            template = np.mean(wfs[local_labels == k, :, :], axis=1)
            (chan_inds,) = np.nonzero(sparsity_mask[l, :])
            assert wfs.shape[2] == chan_inds.size
            main_chan = chan_inds[np.argmax(np.max(np.abs(template), axis=0))]
            main_channels[k + nb_clusters] = main_chan

        if debug:
            # ~ local_labels_with_noise[-noise_size:] = -2
            import matplotlib.pyplot as plt

            # ~ fig, axs = plt.subplots(ncols=3)

            fig, ax = plt.subplots()
            plot_labels_set = np.unique(local_labels_with_noise)
            cmap = plt.colormaps["jet"].resampled(plot_labels_set.size)
            cmap = {k: cmap(l) for l, k in enumerate(plot_labels_set)}
            cmap[-1] = "k"
            cmap[-2] = "b"

            for plot_label in plot_labels_set:
                plot_mask = local_labels_with_noise == plot_label
                color = cmap[plot_label]

                wfs_flat = wfs_and_noise[plot_mask, :, :].swapaxes(1, 2).reshape(np.sum(plot_mask), -1).T
                ax.plot(wfs_flat, color=color, alpha=0.1)
                if plot_label >= 0:
                    ax.plot(wfs_flat.mean(1), color=color, lw=2)

            for c in range(wfs.shape[2]):
                ax.axvline(c * (nbefore + nafter) + nbefore, color="k", ls="-", alpha=0.5)

            if debug_folder is not None:
                fig.savefig(debug_folder / f"auto_split_{label}.png")

            plt.show()

        # remove noise labels
        (mask,) = np.nonzero(peak_labels == label)
        (mask2,) = np.nonzero(local_labels >= 0)
        split_peak_labels[mask[mask2]] = local_labels[mask2] + nb_clusters

        nb_clusters += local_labels.max() + 1

    return split_peak_labels, main_channels


def auto_clean_clustering(
    wfs_arrays,
    sparsity_mask,
    labels,
    peak_labels,
    nbefore,
    nafter,
    channel_distances,
    radius_um=50,
    auto_merge_num_shift=7,
    auto_merge_quantile_limit=0.8,
    ratio_num_channel_intersect=0.8,
):
    """ """
    clean_peak_labels = peak_labels.copy()

    # ~ labels = np.unique(peak_labels)
    # ~ labels = labels[labels >= 0]

    # Check debug
    assert sparsity_mask.shape[0] == labels.shape[0]

    # get main channel on new wfs
    templates = []
    main_channels = []

    for l, label in enumerate(labels):
        wfs = wfs_arrays[label]
        template = np.mean(wfs, axis=0)
        (chan_inds,) = np.nonzero(sparsity_mask[l, :])
        assert wfs.shape[2] == chan_inds.size
        main_chan = chan_inds[np.argmax(np.max(np.abs(template), axis=0))]
        templates.append(template)
        main_channels.append(main_chan)

        # ~ plot_debug = True
        # ~ if plot_debug:
        # ~ import matplotlib.pyplot as plt
        # ~ wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1).T
        # ~ fig, ax = plt.subplots()
        # ~ ax.plot(wfs_flat, color='g', alpha=0.1)
        # ~ ax.plot(template.T.flatten(), color='c', lw=2)
        # ~ plt.show()

    ## Step 1 : auto merge
    auto_merge_list = []
    for l0 in range(len(labels)):
        for l1 in range(l0 + 1, len(labels)):
            label0, label1 = labels[l0], labels[l1]

            main_chan0 = main_channels[l0]
            main_chan1 = main_channels[l1]

            if channel_distances[main_chan0, main_chan1] > radius_um:
                continue

            (channel_inds0,) = np.nonzero(sparsity_mask[l0, :])
            (channel_inds1,) = np.nonzero(sparsity_mask[l1, :])

            intersect_chans = np.intersect1d(channel_inds0, channel_inds1)
            # union_chans = np.union1d(channel_inds0, channel_inds1)

            # we use
            (radius_chans,) = np.nonzero(
                (channel_distances[main_chan0, :] <= radius_um) | (channel_distances[main_chan1, :] <= radius_um)
            )
            if radius_chans.size < (intersect_chans.size * ratio_num_channel_intersect):
                # ~ print('WARNING INTERSECT')
                # ~ print(intersect_chans.size, radius_chans.size, used_chans.size)
                continue

            used_chans = np.intersect1d(radius_chans, intersect_chans)

            if used_chans.size == 0:
                continue

            wfs0 = wfs_arrays[label0]
            wfs0 = wfs0[:, :, np.isin(channel_inds0, used_chans)]
            wfs1 = wfs_arrays[label1]
            wfs1 = wfs1[:, :, np.isin(channel_inds1, used_chans)]

            # TODO : remove
            assert wfs0.shape[2] == wfs1.shape[2]

            equal, shift = check_equal_template_with_distribution_overlap(
                wfs0, wfs1, num_shift=auto_merge_num_shift, quantile_limit=auto_merge_quantile_limit, return_shift=True
            )

            if equal:
                auto_merge_list.append((l0, l1, shift))

            # DEBUG plot
            # ~ plot_debug = debug
            # ~ plot_debug = True
            plot_debug = False
            # ~ plot_debug = equal
            if plot_debug:
                import matplotlib.pyplot as plt

                wfs_flat0 = wfs0.swapaxes(1, 2).reshape(wfs0.shape[0], -1).T
                wfs_flat1 = wfs1.swapaxes(1, 2).reshape(wfs1.shape[0], -1).T
                fig, ax = plt.subplots()
                ax.plot(wfs_flat0, color="g", alpha=0.1)
                ax.plot(wfs_flat1, color="r", alpha=0.1)

                ax.plot(np.mean(wfs_flat0, axis=1), color="c", lw=2)
                ax.plot(np.mean(wfs_flat1, axis=1), color="m", lw=2)

                for c in range(len(used_chans)):
                    ax.axvline(c * (nbefore + nafter) + nbefore, color="k", ls="--")
                ax.set_title(
                    f"label0={label0} label1={label1} equal{equal} shift{shift} \n chans intersect{intersect_chans.size} radius{radius_chans.size} used{used_chans.size}"
                )
                plt.show()

    # ~ print('auto_merge_list', auto_merge_list)
    # merge in reverse order because of shift accumulation
    peak_sample_shifts = np.zeros(peak_labels.size, dtype="int64")
    for l0, l1, shift in auto_merge_list[::-1]:
        label0, label1 = labels[l0], labels[l1]
        (inds,) = np.nonzero(peak_labels == label1)
        clean_peak_labels[inds] = label0
        peak_sample_shifts[inds] += shift

    # update label list
    labels_clean = np.unique(clean_peak_labels)
    labels_clean = labels_clean[labels_clean >= 0]

    # Step 2 : remove none aligner units
    # some unit have a secondary peak that can be detected
    # lets remove the secondary template
    # to avoid recomputing template this is done on the original list
    auto_trash_list = []
    auto_trash_misalignment_shift = auto_merge_num_shift + 1
    for l, label in enumerate(labels):
        if label not in labels_clean:
            continue

        template = templates[l]
        main_chan = main_channels[l]
        (chan_inds,) = np.nonzero(sparsity_mask[l, :])
        max_ind = list(chan_inds).index(main_chan)
        sample_max = np.argmax(np.abs(template[:, max_ind]))

        not_aligned = np.abs(sample_max - nbefore) >= auto_trash_misalignment_shift
        if not_aligned:
            auto_trash_list.append(label)

        # ~ plot_debug = not_aligned
        # ~ plot_debug = True
        plot_debug = False
        # ~ plot_debug = label in (23, )
        if plot_debug:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(nrows=2)
            ax = axs[0]
            wfs_flat = template.T.flatten()
            ax.plot(wfs_flat)
            for c in range(template.shape[1]):
                ax.axvline(c * (nbefore + nafter) + nbefore, color="k", ls="--")
                if c == max_ind:
                    ax.axvline(c * (nbefore + nafter) + sample_max, color="b")

            ax.set_title(
                f"label={label}  not_aligned{not_aligned} chan_inds={chan_inds} chan={chan_inds[max_ind]} max_ind{max_ind}"
            )
            ax = axs[1]
            ax.plot(template[:, max_ind])
            ax.axvline(sample_max)
            ax.axvline(nbefore, color="r", ls="--")
            plt.show()

    # ~ print('auto_trash_list', auto_trash_list)
    for label in auto_trash_list:
        (inds,) = np.nonzero(clean_peak_labels == label)
        clean_peak_labels[inds] = -label

    return clean_peak_labels, peak_sample_shifts


def remove_duplicates(
    wfs_arrays, noise_levels, peak_labels, num_samples, num_chans, cosine_threshold=0.975, sparsify_threshold=0.99
):
    num_templates = len(wfs_arrays.keys())
    templates = np.zeros((num_templates, num_samples, num_chans), dtype=np.float32)

    # All of this is to sparsity the templates
    for t, wfs in wfs_arrays.items():
        templates[t] = np.median(wfs, axis=0)

        is_silent = templates[t].std(0) < 0.1 * noise_levels
        templates[t, :, is_silent] = 0

        channel_norms = np.linalg.norm(templates[t], axis=0) ** 2
        total_norm = np.linalg.norm(templates[t]) ** 2

        idx = np.argsort(channel_norms)[::-1]
        explained_norms = np.cumsum(channel_norms[idx] / total_norm)
        channel = np.searchsorted(explained_norms, sparsify_threshold)
        active_channels = np.sort(idx[:channel])
        templates[t, :, idx[channel:]] = 0

    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(templates.reshape(num_templates, -1))

    # Set the diagonal to -1
    for i in range(num_templates):
        similarities[i, i] = -1

    similar_templates = np.nonzero(similarities > cosine_threshold)

    new_labels = peak_labels.copy()

    labels = np.unique(new_labels)
    labels = labels[labels >= 0]

    for x, y in zip(similar_templates[0], similar_templates[1]):
        mask = new_labels == y
        new_labels[mask] = x

    labels = np.unique(new_labels)
    labels = labels[labels >= 0]

    return labels, new_labels


def detect_mixtures(templates, method_kwargs={}, job_kwargs={}, tmp_folder=None, multiple_passes=False):

    from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
    from spikeinterface.core import BinaryRecordingExtractor, NumpyRecording, SharedMemoryRecording
    import os

    job_kwargs = fix_job_kwargs(job_kwargs)

    templates_array = templates.get_dense_templates()

    nb_templates = len(templates_array)
    duration = templates.nbefore + templates.nafter

    fs = templates.sampling_frequency
    num_chans = len(templates.channel_ids)
    norms = np.linalg.norm(templates_array, axis=(1, 2))
    margin = max(templates.nbefore, templates.nafter)
    tmp_filename = None
    zdata = np.hstack(((templates_array, np.zeros((nb_templates, margin, num_chans)))))
    zdata = zdata.reshape(nb_templates * (duration + margin), num_chans)
    blank = np.zeros((margin, num_chans), dtype=zdata.dtype)
    zdata = np.vstack((blank, zdata, blank))

    if tmp_folder is not None:
        tmp_folder.mkdir(parents=True, exist_ok=True)
        tmp_filename = tmp_folder / "tmp.raw"
        f = open(tmp_filename, "wb")
        f.write(zdata.flatten())
        f.close()
        recording = BinaryRecordingExtractor(
            tmp_filename, num_channels=num_chans, sampling_frequency=fs, dtype=zdata.dtype
        )
    else:
        recording = NumpyRecording(zdata, sampling_frequency=fs)
        recording = SharedMemoryRecording.from_recording(recording)

    recording = recording.set_probe(templates.probe)
    recording.annotate(is_filtered=True)

    local_params = method_kwargs.copy()
    amplitudes = [0.95, 1.05]

    local_params.update({"templates": templates, "amplitudes": amplitudes})

    unit_ids = templates.unit_ids

    ignore_inds = []
    similar_templates = [[], []]

    keep_searching = True

    DEBUG = False
    while keep_searching:

        keep_searching = False

        for i in list(set(range(nb_templates)).difference(ignore_inds)):

            ## Could be speed up by only computing the values for templates that are
            ## nearby

            t_start = i * (duration + margin)
            t_stop = margin + (i + 1) * (duration + margin)

            sub_recording = recording.frame_slice(t_start, t_stop)
            local_params.update({"ignore_inds": ignore_inds + [i]})
            spikes, computed = find_spikes_from_templates(
                sub_recording, method="circus-omp-svd", method_kwargs=local_params, extra_outputs=True, **job_kwargs
            )
            local_params.update(
                {
                    "overlaps": computed["overlaps"],
                    "normed_templates": computed["normed_templates"],
                    "norms": computed["norms"],
                    "temporal": computed["temporal"],
                    "spatial": computed["spatial"],
                    "singular": computed["singular"],
                    "units_overlaps": computed["units_overlaps"],
                    "unit_overlaps_indices": computed["unit_overlaps_indices"],
                }
            )
            valid = (spikes["sample_index"] >= 0) * (spikes["sample_index"] < duration + 2 * margin)

            if np.sum(valid) > 0:
                ref_norm = norms[i]

                j = spikes[valid]["cluster_index"][0]
                sum = templates_array[j]
                for k in range(1, np.sum(valid)):
                    j = spikes[valid]["cluster_index"][k]
                    a = spikes[valid]["amplitude"][k]
                    sum += a * templates_array[j]

                tgt_norm = np.linalg.norm(sum)
                ratio = tgt_norm / ref_norm
                if (amplitudes[0] < ratio) and (ratio < amplitudes[1]):
                    if multiple_passes:
                        keep_searching = True
                    if np.sum(valid) == 1:
                        ignore_inds += [i]
                        similar_templates[1] += [unit_ids[i]]
                        similar_templates[0] += [unit_ids[j]]
                    elif np.sum(valid) > 1:
                        similar_templates[0] += [-1]
                        ignore_inds += [i]
                        similar_templates[1] += [unit_ids[i]]

                    if DEBUG:
                        import pylab as plt

                        fig, axes = plt.subplots(1, 2)
                        from spikeinterface.widgets import plot_traces

                        # plot_traces(sub_recording, ax=axes[0])
                        axes[1].plot(templates_array[i].flatten(), label=f"{ref_norm}")
                        axes[1].plot(sum.flatten(), label=f"{tgt_norm}")
                        axes[1].legend()
                        plt.show()
                        print(i, spikes[valid]["cluster_index"], spikes[valid]["amplitude"])

    del recording, sub_recording, local_params, templates
    if tmp_filename is not None:
        os.remove(tmp_filename)

    return similar_templates


def remove_duplicates_via_matching(
    templates, peak_labels, method_kwargs={}, job_kwargs={}, tmp_folder=None, multiple_passes=False
):

    similar_templates = detect_mixtures(
        templates, method_kwargs, job_kwargs, tmp_folder=tmp_folder, multiple_passes=multiple_passes
    )
    new_labels = peak_labels.copy()

    labels = np.unique(new_labels)
    labels = labels[labels >= 0]

    for x, y in zip(similar_templates[0], similar_templates[1]):
        mask = new_labels == y
        new_labels[mask] = x

    labels = np.unique(new_labels)
    labels = labels[labels >= 0]

    return labels, new_labels


def remove_duplicates_via_dip(wfs_arrays, peak_labels, dip_threshold=1, cosine_threshold=None):
    import sklearn

    from spikeinterface.sortingcomponents.clustering.isocut5 import isocut5

    new_labels = peak_labels.copy()

    keep_merging = True

    fused = {}
    templates = {}
    similarities = {}
    diptests = {}
    cosine = 0

    for i in wfs_arrays.keys():
        fused[i] = [i]

    while keep_merging:
        min_dip = np.inf
        to_merge = None
        labels = np.unique(new_labels)
        labels = labels[labels >= 0]

        for i in labels:
            if len(fused[i]) > 1:
                all_data_i = np.vstack([wfs_arrays[c] for c in fused[i]])
            else:
                all_data_i = wfs_arrays[i]
            n_i = len(all_data_i)
            if n_i > 0:
                if i in templates:
                    t_i = templates[i]
                else:
                    t_i = np.median(all_data_i, axis=0).flatten()
                    templates[i] = t_i
                data_i = all_data_i.reshape(n_i, -1)

                if i not in similarities:
                    similarities[i] = {}

                if i not in diptests:
                    diptests[i] = {}

                for j in labels[i + 1 :]:
                    if len(fused[j]) > 1:
                        all_data_j = np.vstack([wfs_arrays[c] for c in fused[j]])
                    else:
                        all_data_j = wfs_arrays[j]
                    n_j = len(all_data_j)
                    if n_j > 0:
                        if j in templates:
                            t_j = templates[j]
                        else:
                            t_j = np.median(all_data_j, axis=0).flatten()
                            templates[j] = t_j

                        if cosine_threshold is not None:
                            if j in similarities[i]:
                                cosine = similarities[i][j]
                            else:
                                cosine = sklearn.metrics.pairwise.cosine_similarity(
                                    t_i[np.newaxis, :], t_j[np.newaxis, :]
                                )[0][0]
                                similarities[i][j] = cosine

                        if cosine_threshold is None or cosine > cosine_threshold:
                            if j in diptests[i]:
                                diptest = diptests[i][j]
                            else:
                                data_j = all_data_j.reshape(n_j, -1)
                                v = t_i - t_j
                                pr_i = np.dot(data_i, v)
                                pr_j = np.dot(data_j, v)
                                diptest, _ = isocut5(np.concatenate((pr_i, pr_j)))
                                diptests[i][j] = diptest

                            if diptest < min_dip:
                                min_dip = diptest
                                to_merge = [i, j]

        if min_dip < dip_threshold:
            fused[to_merge[0]] += [to_merge[1]]
            mask = new_labels == to_merge[1]
            new_labels[mask] = to_merge[0]
            templates.pop(to_merge[0])
            similarities.pop(to_merge[0])
            diptests.pop(to_merge[0])
        else:
            keep_merging = False

    labels = np.unique(new_labels)
    labels = labels[labels >= 0]

    return labels, new_labels
