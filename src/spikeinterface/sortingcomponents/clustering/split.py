from pathlib import Path
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from hdbscan import HDBSCAN

import numpy as np

from spikeinterface.core.job_tools import get_poolexecutor


def split_clusters(
    labels,
    recording,
    feature_folder,
    method="hdbscan_on_local_pca",
    method_kwargs={},
    recursive=False,
    returns_split_count=False,
    n_jobs=1,
    mp_context="fork",
    max_threads_per_process=1,
    progress_bar=True,
):
    """
    Run recusrsively or not in a multi process pool a local split method.

    Parameters
    ----------
    labels

    feature_folder

    n_jobs=1


    Returns
    -------
    new_labels


    """
    feature_folder = Path(feature_folder)

    original_labels = labels
    labels = labels.copy()
    split_count = np.zeros(labels.size, dtype=int)

    Executor = get_poolexecutor(n_jobs)

    with Executor(
        max_workers=n_jobs,
        initializer=split_worker_init,
        mp_context=get_context(mp_context),
        initargs=(recording, feature_folder, original_labels, method, method_kwargs, max_threads_per_process),
    ) as pool:
        labels_set = np.setdiff1d(labels, [-1])
        current_max_label = np.max(labels_set) + 1

        jobs = []
        for label in labels_set:
            peak_indices = np.flatnonzero(labels == label)
            if peak_indices.size > 0:
                jobs.append(pool.submit(split_function_wrapper, peak_indices))

        if progress_bar:
            iterator = tqdm(jobs, desc=f"split_clusters with {method}", total=len(labels_set))
        else:
            iterator = jobs

        for res in iterator:
            is_split, local_labels, peak_indices = res.result()
            if not is_split:
                continue

            mask = local_labels >= 0
            labels[peak_indices[mask]] = local_labels[mask] + current_max_label
            labels[peak_indices[~mask]] = local_labels[~mask]

            split_count[peak_indices] += 1

            current_max_label += np.max(local_labels[mask]) + 1

            if recursive:
                new_labels_set = np.setdiff1d(labels[peak_indices], [-1])
                for label in new_labels_set:
                    peak_indices = np.flatnonzero(labels == label)
                    if peak_indices.size > 0:
                        jobs.append(pool.submit(split_function_wrapper, peak_indices))
                        if progress_bar:
                            iterator.total += 1

    if returns_split_count:
        return labels, split_count
    else:
        return labels


global _ctx


def split_worker_init(recording, feature_folder, original_labels, method, method_kwargs, max_threads_per_process):
    global _ctx
    _ctx = {}

    _ctx["recording"] = recording
    _ctx["feature_folder"] = feature_folder
    _ctx["original_labels"] = original_labels
    _ctx["method"] = method
    _ctx["method_kwargs"] = method_kwargs
    _ctx["method_class"] = split_methods_dict[method]
    _ctx["max_threads_per_process"] = max_threads_per_process

    features = {}
    for file in feature_folder.glob("*.npy"):
        name = file.stem
        if name == "peaks":
            # load in memory
            _ctx["peaks"] = np.load(file, mmap_mode="r")
        else:
            # memmap load
            features[name] = np.load(file, mmap_mode="r")
    _ctx["features"] = features


def split_function_wrapper(peak_indices):
    global _ctx
    with threadpool_limits(limits=_ctx["max_threads_per_process"]):
        is_split, local_labels = _ctx["method_class"].split(
            peak_indices, _ctx["peaks"], _ctx["features"], **_ctx["method_kwargs"]
        )
    return is_split, local_labels, peak_indices


class HdbscanOnLocalPca:
    name = "hdbscan_on_local_pca"

    def split(
        peak_indices,
        peaks,
        features,
        neighbours_mask=None,
        waveforms_sparse_mask=None,
        min_size_split=25,
        min_cluster_size=25,
        min_samples=25,
        n_pca_features=2,
    ):
        local_labels = np.zeros(peak_indices.size, dtype=np.int64)

        sparse_wfs = features["sparse_wfs"]
        assert waveforms_sparse_mask is not None

        # target channel subset is done intersect local channels + neighbours
        local_chans = np.unique(peaks["channel_index"][peak_indices])
        target_channels = np.flatnonzero(np.all(neighbours_mask[local_chans, :], axis=0))

        aligned_features, dont_have_channels = aggregate_sparse_features(
            peaks, peak_indices, sparse_wfs, waveforms_sparse_mask, target_channels
        )

        local_labels[dont_have_channels] = -2
        kept = np.flatnonzero(~dont_have_channels)
        if kept.size < min_size_split:
            return False, None

        flatten_features = aligned_features[kept].reshape(kept.size, -1)
        pca_features = PCA(n_pca_features, whiten=True).fit_transform(flatten_features)

        clust = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clust.fit(pca_features)
        possible_labels = clust.labels_

        is_split = np.setdiff1d(possible_labels, [-1]).size > 1

        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt

            label_sets = np.setdiff1d(possible_labels, [-1])
            colors = plt.get_cmap("tab10", len(label_sets))
            colors = {k: colors(i) for i, k in enumerate(label_sets)}
            colors[-1] = "k"
            fix, axs = plt.subplots(nrows=2)

            flatten_wfs = aligned_features.swapaxes(1, 2).reshape(aligned_features.shape[0], -1)

            for k in np.unique(possible_labels):
                mask = possible_labels == k
                ax = axs[0]
                ax.scatter(pca_features[:, 0][mask], pca_features[:, 1][mask], s=5, color=colors[k])

                ax = axs[1]
                ax.plot(flatten_wfs[mask].T, color=colors[k], alpha=0.5)

            plt.show()

        if not is_split:
            return is_split, None

        local_labels[kept] = possible_labels

        return is_split, local_labels


def aggregate_sparse_features(peaks, peak_indices, sparse_feature, sparse_mask, target_channels):
    """
    Aggregate sparse features that have unaligned channels and realigned then on target_channels


    """
    local_peaks = peaks[peak_indices]

    aligned_features = np.zeros(
        (local_peaks.size, sparse_feature.shape[1], target_channels.size), dtype=sparse_feature.dtype
    )
    dont_have_channels = np.zeros(peak_indices.size, dtype=bool)

    for chan in np.unique(local_peaks["channel_index"]):
        sparse_chans = np.flatnonzero(sparse_mask[chan, :])
        peak_inds = np.flatnonzero(local_peaks["channel_index"] == chan)
        if np.all(np.in1d(target_channels, sparse_chans)):
            # peaks feature channel have all target_channels
            source_chans = np.flatnonzero(np.in1d(sparse_chans, target_channels))
            aligned_features[peak_inds, :, :] = sparse_feature[peak_indices[peak_inds], :, :][:, :, source_chans]
        else:
            # some channel are missing, peak are not removde
            dont_have_channels[peak_inds] = True

    return aligned_features, dont_have_channels


split_methods_list = [
    HdbscanOnLocalPca,
]
split_methods_dict = {e.name: e for e in split_methods_list}
