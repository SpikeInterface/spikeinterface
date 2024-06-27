from __future__ import annotations

from multiprocessing import get_context
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


import numpy as np

from spikeinterface.core.job_tools import get_poolexecutor, fix_job_kwargs

from .tools import aggregate_sparse_features, FeaturesLoader

try:
    import numba
    from .isocut5 import isocut5
except:
    pass  # isocut requires numba

# important all DEBUG and matplotlib are left in the code intentionally


def split_clusters(
    peak_labels,
    recording,
    features_dict_or_folder,
    method="hdbscan_on_local_pca",
    method_kwargs={},
    recursive=False,
    recursive_depth=None,
    returns_split_count=False,
    **job_kwargs,
):
    """
    Run recusrsively (or not) in a multi process pool a local split method.

    Parameters
    ----------
    peak_labels: numpy.array
        Peak label before split
    recording: Recording
        Recording object
    features_dict_or_folder: dict or folder
        A dictionary of features precomputed with peak_pipeline or a folder containing npz file for features
    method: str, default: "hdbscan_on_local_pca"
        The method name
    method_kwargs: dict, default: dict()
        The method option
    recursive: bool, default: False
        Recursive or not
    recursive_depth: None or int, default: None
        If recursive=True, then this is the max split per spikes
    returns_split_count: bool, default: False
        Optionally return  the split count vector. Same size as labels

    Returns
    -------
    new_labels: numpy.ndarray
        The labels of peaks after split.
    split_count: numpy.ndarray
        Optionally returned
    """

    job_kwargs = fix_job_kwargs(job_kwargs)
    n_jobs = job_kwargs["n_jobs"]
    mp_context = job_kwargs.get("mp_context", None)
    progress_bar = job_kwargs["progress_bar"]
    max_threads_per_process = job_kwargs.get("max_threads_per_process", 1)

    original_labels = peak_labels
    peak_labels = peak_labels.copy()
    split_count = np.zeros(peak_labels.size, dtype=int)

    Executor = get_poolexecutor(n_jobs)

    with Executor(
        max_workers=n_jobs,
        initializer=split_worker_init,
        mp_context=get_context(method=mp_context),
        initargs=(recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_process),
    ) as pool:
        labels_set = np.setdiff1d(peak_labels, [-1])
        current_max_label = np.max(labels_set) + 1

        jobs = []
        for label in labels_set:
            peak_indices = np.flatnonzero(peak_labels == label)
            if peak_indices.size > 0:
                jobs.append(pool.submit(split_function_wrapper, peak_indices, 1))

        if progress_bar:
            iterator = tqdm(jobs, desc=f"split_clusters with {method}", total=len(labels_set))
        else:
            iterator = jobs

        for res in iterator:
            is_split, local_labels, peak_indices = res.result()
            if not is_split:
                continue

            mask = local_labels >= 0
            peak_labels[peak_indices[mask]] = local_labels[mask] + current_max_label
            peak_labels[peak_indices[~mask]] = local_labels[~mask]

            split_count[peak_indices] += 1

            current_max_label += np.max(local_labels[mask]) + 1

            if recursive:
                recursion_level = np.max(split_count[peak_indices])
                if recursive_depth is not None:
                    # stop reccursivity when recursive_depth is reach
                    extra_ball = recursion_level < recursive_depth
                else:
                    # reccurssive always
                    extra_ball = True

                if extra_ball:
                    new_labels_set = np.setdiff1d(peak_labels[peak_indices], [-1])
                    for label in new_labels_set:
                        peak_indices = np.flatnonzero(peak_labels == label)
                        if peak_indices.size > 0:
                            jobs.append(pool.submit(split_function_wrapper, peak_indices, recursion_level))
                            if progress_bar:
                                iterator.total += 1

    if returns_split_count:
        return peak_labels, split_count
    else:
        return peak_labels


global _ctx


def split_worker_init(
    recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_process
):
    global _ctx
    _ctx = {}

    _ctx["recording"] = recording
    features_dict_or_folder
    _ctx["original_labels"] = original_labels
    _ctx["method"] = method
    _ctx["method_kwargs"] = method_kwargs
    _ctx["method_class"] = split_methods_dict[method]
    _ctx["max_threads_per_process"] = max_threads_per_process
    _ctx["features"] = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)
    _ctx["peaks"] = _ctx["features"]["peaks"]


def split_function_wrapper(peak_indices, recursion_level):
    global _ctx
    with threadpool_limits(limits=_ctx["max_threads_per_process"]):
        is_split, local_labels = _ctx["method_class"].split(
            peak_indices, _ctx["peaks"], _ctx["features"], recursion_level, **_ctx["method_kwargs"]
        )
    return is_split, local_labels, peak_indices


class LocalFeatureClustering:
    """
    This method is a refactorized mix  between:
       * old tridesclous code
       * "herding_split()" in DART/spikepsvae by Charlie Windolf

    The idea simple :
     * agregate features (svd or even waveforms) with sparse channel.
     * run a local feature reduction (pca or svd)
     * try a new split (hdscan or isocut5)
    """

    name = "local_feature_clustering"

    @staticmethod
    def split(
        peak_indices,
        peaks,
        features,
        recursion_level=1,
        clusterer="hdbscan",
        feature_name="sparse_tsvd",
        neighbours_mask=None,
        waveforms_sparse_mask=None,
        clusterer_kwargs={"min_cluster_size": 25},
        min_size_split=25,
        n_pca_features=2,
        scale_n_pca_by_depth=False,
        minimum_common_channels=2,
    ):
        local_labels = np.zeros(peak_indices.size, dtype=np.int64)

        # can be sparse_tsvd or sparse_wfs
        sparse_features = features[feature_name]

        assert waveforms_sparse_mask is not None

        # target channel subset is done intersect local channels + neighbours
        local_chans = np.unique(peaks["channel_index"][peak_indices])

        target_channels = np.flatnonzero(np.all(neighbours_mask[local_chans, :], axis=0))

        # TODO fix this a better way, this when cluster have too few overlapping channels
        if target_channels.size < minimum_common_channels:
            return False, None

        aligned_wfs, dont_have_channels = aggregate_sparse_features(
            peaks, peak_indices, sparse_features, waveforms_sparse_mask, target_channels
        )

        local_labels[dont_have_channels] = -2
        kept = np.flatnonzero(~dont_have_channels)

        if kept.size < min_size_split:
            return False, None

        aligned_wfs = aligned_wfs[kept, :, :]

        flatten_features = aligned_wfs.reshape(aligned_wfs.shape[0], -1)

        if flatten_features.shape[1] > n_pca_features:
            from sklearn.decomposition import PCA

            if scale_n_pca_by_depth:
                # tsvd = TruncatedSVD(n_pca_features * recursion_level)
                tsvd = PCA(n_pca_features * recursion_level, whiten=True)
            else:
                # tsvd = TruncatedSVD(n_pca_features)
                tsvd = PCA(n_pca_features, whiten=True)
            final_features = tsvd.fit_transform(flatten_features)
        else:
            final_features = flatten_features

        if clusterer == "hdbscan":
            from hdbscan import HDBSCAN

            clust = HDBSCAN(**clusterer_kwargs)
            clust.fit(final_features)
            possible_labels = clust.labels_
            is_split = np.setdiff1d(possible_labels, [-1]).size > 1
        elif clusterer == "isocut5":
            min_cluster_size = clusterer_kwargs["min_cluster_size"]
            dipscore, cutpoint = isocut5(final_features[:, 0])
            possible_labels = np.zeros(final_features.shape[0])
            if dipscore > 1.5:
                mask = final_features[:, 0] > cutpoint
                if np.sum(mask) > min_cluster_size and np.sum(~mask):
                    possible_labels[mask] = 1
                is_split = np.setdiff1d(possible_labels, [-1]).size > 1
            else:
                is_split = False
        else:
            raise ValueError(f"wrong clusterer {clusterer}")

        # DEBUG = True
        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt

            labels_set = np.setdiff1d(possible_labels, [-1])
            colors = plt.colormaps["tab10"].resampled(len(labels_set))
            colors = {k: colors(i) for i, k in enumerate(labels_set)}
            colors[-1] = "k"
            fix, axs = plt.subplots(nrows=2)

            flatten_wfs = aligned_wfs.swapaxes(1, 2).reshape(aligned_wfs.shape[0], -1)

            sl = slice(None, None, 10)
            for k in np.unique(possible_labels):
                mask = possible_labels == k
                ax = axs[0]
                ax.scatter(final_features[:, 0][mask][sl], final_features[:, 1][mask][sl], s=5, color=colors[k])

                ax = axs[1]
                ax.plot(flatten_wfs[mask][sl].T, color=colors[k], alpha=0.5)

            axs[0].set_title(f"{clusterer} {is_split} {peak_indices[0]} {np.unique(possible_labels)}")
            plt.show()

        if not is_split:
            return is_split, None

        local_labels[kept] = possible_labels

        return is_split, local_labels


split_methods_list = [
    LocalFeatureClustering,
]
split_methods_dict = {e.name: e for e in split_methods_list}
