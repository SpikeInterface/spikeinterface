from __future__ import annotations

# """Sorting components: clustering"""
from pathlib import Path
import time
import random
import string


import numpy as np

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False


from spikeinterface.core import (
    get_global_tmp_folder,
    get_channel_distances,
    get_random_data_chunks,
    extract_waveforms_to_buffers,
)
from .clustering_tools import auto_clean_clustering, auto_split_clustering


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
        "waveform_mode": "shared_memory",
        "tmp_folder": None,
        "ms_before": 1.5,
        "ms_after": 2.5,
        "noise_size": 300,
        "min_spike_on_channel": 5,
        "stop_explore_percent": 0.05,
        "min_cluster_size": 10,
        "radius_um": 50.0,
        "n_components_by_channel": 4,
        "auto_merge_num_shift": 3,
        "auto_merge_quantile_limit": 0.8,
        "ratio_num_channel_intersect": 0.5,
        # ~ 'auto_trash_misalignment_shift' : 4,
        "job_kwargs": {"n_jobs": -1, "chunk_memory": "10M", "progress_bar": True},
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        assert HAVE_HDBSCAN, "sliding_hdbscan clustering need hdbscan to be installed"
        params = cls._check_params(recording, peaks, params)
        wfs_arrays, sparsity_mask, noise = cls._initialize_folder(recording, peaks, params)
        peak_labels = cls._find_clusters(recording, peaks, wfs_arrays, sparsity_mask, noise, params)

        wfs_arrays2, sparsity_mask2 = cls._prepare_clean(
            recording, peaks, wfs_arrays, sparsity_mask, peak_labels, params
        )

        clean_peak_labels, peak_sample_shifts = cls._clean_cluster(
            recording, peaks, wfs_arrays2, sparsity_mask2, peak_labels, params
        )

        labels = np.unique(clean_peak_labels)
        labels = labels[labels >= 0]

        return labels, peak_labels

    @classmethod
    def _check_params(cls, recording, peaks, params):
        d = params
        params2 = params.copy()

        tmp_folder = params["tmp_folder"]
        if d["waveform_mode"] == "memmap":
            if tmp_folder is None:
                name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                tmp_folder = get_global_tmp_folder() / f"SlidingHdbscanClustering_{name}"
            else:
                tmp_folder = Path(tmp_folder)
            tmp_folder.mkdir()
            params2["tmp_folder"] = tmp_folder
        elif d["waveform_mode"] == "shared_memory":
            assert tmp_folder is None, "temp_folder must be None for shared_memory"
        else:
            raise ValueError("shared_memory")

        return params2

    @classmethod
    def _initialize_folder(cls, recording, peaks, params):
        d = params
        tmp_folder = params["tmp_folder"]

        num_chans = recording.channel_ids.size

        # important sparsity is 2 times radius sparsity because closest channel will be 1 time radius
        chan_distances = get_channel_distances(recording)
        sparsity_mask = np.zeros((num_chans, num_chans), dtype="bool")
        for c in range(num_chans):
            (chans,) = np.nonzero(chan_distances[c, :] <= (2 * d["radius_um"]))
            sparsity_mask[c, chans] = True

        # create a new peak vector to extract waveforms
        dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
        peaks2 = np.zeros(peaks.size, dtype=dtype)
        peaks2["sample_index"] = peaks["sample_index"]
        peaks2["unit_index"] = peaks["channel_index"]
        peaks2["segment_index"] = peaks["segment_index"]

        fs = recording.get_sampling_frequency()
        dtype = recording.get_dtype()

        nbefore = int(d["ms_before"] * fs / 1000.0)
        nafter = int(d["ms_after"] * fs / 1000.0)

        if tmp_folder is None:
            wf_folder = None
        else:
            wf_folder = tmp_folder / "waveforms"
            wf_folder.mkdir()

        ids = np.arange(num_chans, dtype="int64")
        wfs_arrays = extract_waveforms_to_buffers(
            recording,
            peaks2,
            ids,
            nbefore,
            nafter,
            mode=d["waveform_mode"],
            return_scaled=False,
            folder=wf_folder,
            dtype=dtype,
            sparsity_mask=sparsity_mask,
            copy=(d["waveform_mode"] == "shared_memory"),
            **d["job_kwargs"],
        )

        # noise
        noise = get_random_data_chunks(
            recording,
            return_scaled=False,
            num_chunks_per_segment=d["noise_size"],
            chunk_size=nbefore + nafter,
            concatenated=False,
            seed=None,
        )
        noise = np.stack(noise, axis=0)

        return wfs_arrays, sparsity_mask, noise

    @classmethod
    def _find_clusters(cls, recording, peaks, wfs_arrays, sparsity_mask, noise, d):

        import sklearn.decomposition

        num_chans = recording.get_num_channels()
        fs = recording.get_sampling_frequency()
        nbefore = int(d["ms_before"] * fs / 1000.0)
        nafter = int(d["ms_after"] * fs / 1000.0)

        possible_channel_inds = np.unique(peaks["channel_index"])

        # channel neighborhood
        chan_distances = get_channel_distances(recording)
        closest_channels = []
        for c in range(num_chans):
            (chans,) = np.nonzero(chan_distances[c, :] <= d["radius_um"])
            chans = np.intersect1d(possible_channel_inds, chans)
            closest_channels.append(chans)

        peak_labels = np.zeros(peaks.size, dtype="int64")

        # create amplitudes percentile vector
        # this help to explore channel starting with high amplitudes
        chan_amps = np.zeros(num_chans, dtype="float64")
        remain_count = np.zeros(num_chans, dtype="int64")
        remain_percent = np.zeros(num_chans, dtype="float64")
        total_count = np.zeros(num_chans, dtype="int64")
        for chan_ind in range(num_chans):
            total_count[chan_ind] = np.sum(peaks["channel_index"] == chan_ind)

        # this force compute compute at forst loop
        prev_local_chan_inds = np.arange(num_chans, dtype="int64")

        actual_label = 1

        while True:
            # update ampltiude percentile and count peak by channel
            for chan_ind in prev_local_chan_inds:
                if total_count[chan_ind] == 0:
                    continue
                # ~ inds, = np.nonzero(np.isin(peaks['channel_index'], closest_channels[chan_ind]) & (peak_labels==0))
                (inds,) = np.nonzero((peaks["channel_index"] == chan_ind) & (peak_labels == 0))
                if inds.size <= d["min_spike_on_channel"]:
                    chan_amps[chan_ind] = 0.0
                else:
                    amps = np.abs(peaks["amplitude"][inds])
                    chan_amps[chan_ind] = np.percentile(amps, 90)
                remain_count[chan_ind] = inds.size
                remain_percent[chan_ind] = remain_count[chan_ind] / total_count[chan_ind]
                if remain_percent[chan_ind] < d["stop_explore_percent"]:
                    chan_amps[chan_ind] = 0.0

            # get best channel
            if np.all(chan_amps == 0):
                break

            # try fist unexplore and high amplitude
            # local_chan_ind = np.argmax(chan_amps)
            local_chan_ind = np.argmax(chan_amps * remain_percent)
            local_chan_inds = closest_channels[local_chan_ind]

            # take waveforms not label yet for channel in radius
            # ~ t0 = time.perf_counter()
            wfs = []
            local_peak_ind = []
            for chan_ind in local_chan_inds:
                (sel,) = np.nonzero(peaks["channel_index"] == chan_ind)
                (inds,) = np.nonzero(peak_labels[sel] == 0)
                local_peak_ind.append(sel[inds])
                # here a unit is a channel index!!!
                wfs_chan = wfs_arrays[chan_ind]

                # TODO: only for debug, remove later
                assert wfs_chan.shape[0] == sel.size

                (wf_chans,) = np.nonzero(sparsity_mask[chan_ind])
                # TODO: only for debug, remove later
                assert np.all(np.isin(local_chan_inds, wf_chans))

                # none label spikes
                wfs_chan = wfs_chan[inds, :, :]
                # only some channels
                wfs_chan = wfs_chan[:, :, np.isin(wf_chans, local_chan_inds)]
                wfs.append(wfs_chan)

            # put noise to enhance clusters
            wfs.append(noise[:, :, local_chan_inds])
            wfs = np.concatenate(wfs, axis=0)
            local_peak_ind = np.concatenate(local_peak_ind, axis=0)
            # ~ t1 = time.perf_counter()
            # ~ print('WFS time',  t1 - t0)

            # reduce dim : PCA
            # ~ t0 = time.perf_counter()
            n = d["n_components_by_channel"]
            local_feature = np.zeros((wfs.shape[0], d["n_components_by_channel"] * len(local_chan_inds)))

            # ~ tsvd = sklearn.decomposition.TruncatedSVD(n_components=n)
            # ~ plot_labels = []
            # ~ for c in range(wfs.shape[2]):
            # ~ local_feature[:, c*n:(c+1)*n] = tsvd.fit_transform(wfs[:, :, c])
            # ~ pca = sklearn.decomposition.PCA(n_components=d['n_components_by_channel'], whiten=True)
            # ~ local_feature = pca.fit_transform(local_feature)

            pca = sklearn.decomposition.TruncatedSVD(n_components=n)
            for c, chan_ind in enumerate(local_chan_inds):
                local_feature[:, c * n : (c + 1) * n] = pca.fit_transform(wfs[:, :, c])

            wfs_flat = wfs.reshape(wfs.shape[0], -1)
            # ~ t1 = time.perf_counter()
            # ~ print('PCA time',  t1 - t0)

            # find some clusters
            # ~ t0 = time.perf_counter()
            clusterer = hdbscan.HDBSCAN(min_cluster_size=d["min_cluster_size"], allow_single_cluster=True, metric="l2")
            all_labels = clusterer.fit_predict(local_feature)
            # ~ t1 = time.perf_counter()
            # ~ print('HDBSCAN time',  t1 - t0)

            # ~ t0 = time.perf_counter()
            local_labels = all_labels[: -noise.shape[0]]
            noise_labels = all_labels[-noise.shape[0] :]

            local_labels_set = np.unique(local_labels)

            num_cluster = np.sum(local_labels_set >= 0)

            if num_cluster > 1:
                # take only the best cluster = best amplitude on central channel
                # other cluster will be taken in a next loop
                ind = local_chan_inds.tolist().index(local_chan_ind)
                peak_values = wfs[: -noise.shape[0], nbefore, ind]
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
                (outlier_inds,) = np.nonzero(
                    (local_labels == -1) & (peaks[local_peak_ind]["channel_index"] == local_chan_ind)
                )
                if outlier_inds.size > 0:
                    peak_labels[local_peak_ind[outlier_inds]] = -1
            elif num_cluster == 1:
                best_label = 0
                final_peak_inds = local_peak_ind[local_labels >= 0]
            else:
                best_label = None
                final_peak_inds = np.array([], dtype="int64")
                # trash all peaks from this channel
                (to_trash_ind,) = np.nonzero(peaks[local_peak_ind]["channel_index"] == local_chan_ind)
                peak_labels[local_peak_ind[to_trash_ind]] = -1

            if best_label is not None:
                if final_peak_inds.size >= d["min_cluster_size"]:
                    peak_labels[final_peak_inds] = actual_label
                else:
                    peak_labels[final_peak_inds] = -actual_label
                actual_label += 1

            # ~ t1 = time.perf_counter()
            # ~ print('label time',  t1 - t0)

            # this force recompute amplitude and count at next loop
            prev_local_chan_inds = local_chan_inds

            # DEBUG plot
            # ~ plot_debug = True
            plot_debug = False

            if plot_debug:
                import matplotlib.pyplot as plt
                import umap

                reducer = umap.UMAP()
                reduce_local_feature_all = reducer.fit_transform(local_feature)
                reduce_local_feature = reduce_local_feature_all[: -noise.shape[0]]
                reduce_local_feature_noise = reduce_local_feature_all[-noise.shape[0] :]

                wfs_no_noise = wfs[: -noise.shape[0]]

                fig, axs = plt.subplots(ncols=3)
                cmap = plt.colormaps["jet"].resampled(np.unique(local_labels).size)
                cmap = {label: cmap(l) for l, label in enumerate(local_labels_set)}
                cmap[-1] = "k"
                for label in local_labels_set:
                    color = cmap[label]
                    ax = axs[0]
                    mask = local_labels == label
                    ax.scatter(reduce_local_feature[mask, 0], reduce_local_feature[mask, 1], color=color)

                    # scatter noise
                    mask_noise = noise_labels == label
                    if np.any(mask_noise):
                        ax.scatter(
                            reduce_local_feature_noise[mask_noise, 0],
                            reduce_local_feature_noise[mask_noise, 1],
                            color=color,
                            marker="*",
                        )

                    ax = axs[1]
                    wfs_flat2 = wfs_no_noise[mask, :, :].swapaxes(1, 2).reshape(np.sum(mask), -1).T
                    ax.plot(wfs_flat2, color=color)
                    if label == best_label:
                        ax.plot(np.mean(wfs_flat2, axis=1), color="m", lw=2)
                    if num_cluster > 1:
                        if outlier_inds.size > 0:
                            wfs_flat2 = wfs_no_noise[outlier_inds, :, :].swapaxes(1, 2).reshape(outlier_inds.size, -1).T
                            ax.plot(wfs_flat2, color="red", ls="--")
                    if num_cluster > 1:
                        ax = axs[2]
                        count, bins = np.histogram(peak_values[mask], bins=35)
                        ax.plot(bins[:-1], count, color=color)
                ax = axs[1]
                for c in range(len(local_chan_inds)):
                    ax.axvline(c * (nbefore + nafter) + nbefore, color="k", ls="--")
                ax.set_title(
                    f"n={local_peak_ind.size} labeled={final_peak_inds.size} chans={local_chan_ind} {local_chan_inds}"
                )

                ax = axs[2]
                (sel,) = np.nonzero((peaks["channel_index"] == local_chan_ind) & (peak_labels == 0))
                count, bins = np.histogram(np.abs(peaks["amplitude"][sel]), bins=200)
                ax.plot(bins[:-1], count, color="k", alpha=0.5)

                plt.show()
            # END DEBUG plot

        peak_labels[peak_labels == 0] = -1

        return peak_labels

    @classmethod
    def _prepare_clean(cls, recording, peaks, wfs_arrays, sparsity_mask, peak_labels, d):
        tmp_folder = d["tmp_folder"]
        if tmp_folder is None:
            wf_folder = None
        else:
            wf_folder = tmp_folder / "waveforms_pre_clean"
            wf_folder.mkdir()

        num_chans = recording.get_num_channels()
        fs = recording.get_sampling_frequency()
        nbefore = int(d["ms_before"] * fs / 1000.0)
        nafter = int(d["ms_after"] * fs / 1000.0)

        possible_channel_inds = np.unique(peaks["channel_index"])
        chan_distances = get_channel_distances(recording)
        closest_channels = []
        for c in range(num_chans):
            (chans,) = np.nonzero(chan_distances[c, :] <= (d["radius_um"]) * 2)
            closest_channels.append(chans)

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        # loop over label take wafevorm from channel and get main channel
        main_channels = []
        for l, label in enumerate(labels):
            wfs, chan_inds = _collect_sparse_waveforms(
                peaks, wfs_arrays, closest_channels, peak_labels, sparsity_mask, label
            )
            template = np.mean(wfs, axis=0)
            main_chan = chan_inds[np.argmax(np.max(np.abs(template), axis=0))]
            main_channels.append(main_chan)

        # extact again waveforms based on new sparsity mask depending on main_chan
        dtype = recording.get_dtype()
        # ~ return_scaled = False
        peak_dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
        keep = peak_labels >= 0
        num_keep = np.sum(keep)
        keep_peak_labels = peak_labels[keep]
        peaks2 = np.zeros(num_keep, dtype=peak_dtype)
        peaks2["sample_index"] = peaks["sample_index"][keep]
        peaks2["segment_index"] = peaks["segment_index"][keep]
        sparsity_mask2 = np.zeros((labels.shape[0], num_chans), dtype="bool")
        for l, label in enumerate(labels):
            main_chan = main_channels[l]
            mask = keep_peak_labels == label
            peaks2["unit_index"][mask] = l
            # here we take a twice radius
            (closest_chans,) = np.nonzero(chan_distances[main_chan, :] <= d["radius_um"] * 2)
            sparsity_mask2[l, closest_chans] = True

        wfs_arrays2 = extract_waveforms_to_buffers(
            recording,
            peaks2,
            labels,
            nbefore,
            nafter,
            mode=d["waveform_mode"],
            return_scaled=False,
            folder=wf_folder,
            dtype=recording.get_dtype(),
            sparsity_mask=sparsity_mask2,
            copy=(d["waveform_mode"] == "shared_memory"),
            **d["job_kwargs"],
        )

        return wfs_arrays2, sparsity_mask2

    @classmethod
    def _clean_cluster(cls, recording, peaks, wfs_arrays2, sparsity_mask2, peak_labels, d):
        fs = recording.get_sampling_frequency()
        nbefore = int(d["ms_before"] * fs / 1000.0)
        nafter = int(d["ms_after"] * fs / 1000.0)

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        chan_locs = recording.get_channel_locations()
        channel_distances = get_channel_distances(recording)

        clean_peak_labels, peak_sample_shifts = auto_clean_clustering(
            wfs_arrays2,
            sparsity_mask2,
            labels,
            peak_labels,
            nbefore,
            nafter,
            channel_distances,
            radius_um=d["radius_um"],
            auto_merge_num_shift=d["auto_merge_num_shift"],
            auto_merge_quantile_limit=d["auto_merge_quantile_limit"],
            ratio_num_channel_intersect=d["ratio_num_channel_intersect"],
        )

        return clean_peak_labels, peak_sample_shifts


def _collect_sparse_waveforms(peaks, wfs_arrays, closest_channels, peak_labels, sparsity_mask, label):
    (inds,) = np.nonzero(peak_labels == label)
    local_peaks = peaks[inds]
    label_chan_inds, count = np.unique(local_peaks["channel_index"], return_counts=True)
    main_chan = label_chan_inds[np.argmax(count)]

    # only main channel sparsity
    wanted_chans = closest_channels[main_chan]
    for chan_ind in label_chan_inds:
        # remove channel non in common
        wanted_chans = np.intersect1d(wanted_chans, closest_channels[chan_ind])
    # print('wanted_chans', wanted_chans)

    wfs = []
    for chan_ind in label_chan_inds:
        (sel,) = np.nonzero(peaks["channel_index"] == chan_ind)

        (inds,) = np.nonzero(peak_labels[sel] == label)

        (wf_chans,) = np.nonzero(sparsity_mask[chan_ind])
        # print('wf_chans', wf_chans)
        # TODO: only for debug, remove later
        assert np.all(np.isin(wanted_chans, wf_chans))
        wfs_chan = wfs_arrays[chan_ind]

        # TODO: only for debug, remove later
        assert wfs_chan.shape[0] == sel.size

        wfs_chan = wfs_chan[inds, :, :]
        # only some channels
        wfs_chan = wfs_chan[:, :, np.isin(wf_chans, wanted_chans)]
        wfs.append(wfs_chan)

    wfs = np.concatenate(wfs, axis=0)

    # TODO DEBUG and check
    assert wanted_chans.shape[0] == wfs.shape[2]

    return wfs, wanted_chans
