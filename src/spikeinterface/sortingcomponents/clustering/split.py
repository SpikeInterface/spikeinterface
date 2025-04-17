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
    method="local_feature_clustering",
    method_kwargs={},
    recursive=False,
    recursive_depth=None,
    returns_split_count=False,
    debug_folder=None,
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
    method: str, default: "local_feature_clustering"
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
    max_threads_per_worker = job_kwargs.get("max_threads_per_worker", 1)

    original_labels = peak_labels
    peak_labels = peak_labels.copy()
    split_count = np.zeros(peak_labels.size, dtype=int)
    recursion_level = 1
    Executor = get_poolexecutor(n_jobs)

    with Executor(
        max_workers=n_jobs,
        initializer=split_worker_init,
        mp_context=get_context(method=mp_context),
        initargs=(recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_worker),
    ) as pool:
        labels_set = np.setdiff1d(peak_labels, [-1])
        current_max_label = np.max(labels_set) + 1
        jobs = []

        if debug_folder is not None:
            if debug_folder.exists():
                import shutil

                shutil.rmtree(debug_folder)
            debug_folder.mkdir(parents=True, exist_ok=True)

        for label in labels_set:
            peak_indices = np.flatnonzero(peak_labels == label)
            if debug_folder is not None:
                sub_folder = str(debug_folder / f"split_{label}")

            else:
                sub_folder = None
            if peak_indices.size > 0:
                jobs.append(pool.submit(split_function_wrapper, peak_indices, recursion_level, sub_folder))

        if progress_bar:
            pbar = tqdm(desc=f"split_clusters with {method}", total=len(labels_set))

        for res in jobs:
            is_split, local_labels, peak_indices, sub_folder = res.result()

            if progress_bar:
                pbar.update(1)

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
                    # stop recursivity when recursive_depth is reach
                    extra_ball = recursion_level < recursive_depth
                else:
                    # recursive always
                    extra_ball = True

                if extra_ball:
                    new_labels_set = np.setdiff1d(peak_labels[peak_indices], [-1])
                    for label in new_labels_set:
                        peak_indices = np.flatnonzero(peak_labels == label)
                        if sub_folder is not None:
                            new_sub_folder = sub_folder + f"_{label}"
                        else:
                            new_sub_folder = None
                        if peak_indices.size > 0:
                            # print('Relaunched', label, len(peak_indices), recursion_level)
                            jobs.append(
                                pool.submit(split_function_wrapper, peak_indices, recursion_level, new_sub_folder)
                            )
                            if progress_bar:
                                pbar.total += 1

        if progress_bar:
            pbar.close()
            del pbar

    if returns_split_count:
        return peak_labels, split_count
    else:
        return peak_labels


global _ctx


def split_worker_init(
    recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_worker
):
    global _ctx
    _ctx = {}

    _ctx["recording"] = recording
    features_dict_or_folder
    _ctx["original_labels"] = original_labels
    _ctx["method"] = method
    _ctx["method_kwargs"] = method_kwargs
    _ctx["method_class"] = split_methods_dict[method]
    _ctx["max_threads_per_worker"] = max_threads_per_worker
    _ctx["features"] = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)
    _ctx["peaks"] = _ctx["features"]["peaks"]


def split_function_wrapper(peak_indices, recursion_level, debug_folder):
    global _ctx
    with threadpool_limits(limits=_ctx["max_threads_per_worker"]):
        is_split, local_labels = _ctx["method_class"].split(
            peak_indices, _ctx["peaks"], _ctx["features"], recursion_level, debug_folder, **_ctx["method_kwargs"]
        )
    return is_split, local_labels, peak_indices, debug_folder


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
        debug_folder=None,
        clusterer="hdbscan",
        clusterer_kwargs={"min_cluster_size": 25, "min_samples": 5},
        feature_name="sparse_tsvd",
        neighbours_mask=None,
        waveforms_sparse_mask=None,
        min_size_split=25,
        n_pca_features=2,
        projection_mode="tsvd",
        minimum_overlap_ratio=0.25,
    ):
        local_labels = np.zeros(peak_indices.size, dtype=np.int64)

        # can be sparse_tsvd or sparse_wfs
        sparse_features = features[feature_name]

        assert waveforms_sparse_mask is not None

        # target channel subset is done intersect local channels + neighbours
        local_chans = np.unique(peaks["channel_index"][peak_indices])

        target_intersection_channels = np.flatnonzero(np.all(neighbours_mask[local_chans, :], axis=0))
        target_union_channels = np.flatnonzero(np.any(neighbours_mask[local_chans, :], axis=0))
        num_intersection = len(target_intersection_channels)
        num_union = len(target_union_channels)

        # TODO fix this a better way, this when cluster have too few overlapping channels
        if (num_intersection / num_union) < minimum_overlap_ratio:
            return False, None

        aligned_wfs, dont_have_channels = aggregate_sparse_features(
            peaks, peak_indices, sparse_features, waveforms_sparse_mask, target_intersection_channels
        )

        local_labels[dont_have_channels] = -2
        kept = np.flatnonzero(~dont_have_channels)

        if kept.size < min_size_split:
            return False, None

        aligned_wfs = aligned_wfs[kept, :, :]
        flatten_features = aligned_wfs.reshape(aligned_wfs.shape[0], -1)

        is_split = False

        if isinstance(n_pca_features, float):
            assert 0 < n_pca_features < 1, "n_components should be in ]0, 1["
            nb_dimensions = min(flatten_features.shape[0], flatten_features.shape[1])
            if projection_mode == "pca":
                from sklearn.decomposition import PCA

                tsvd = PCA(nb_dimensions, whiten=True)
            elif projection_mode == "tsvd":
                from sklearn.decomposition import TruncatedSVD

                tsvd = TruncatedSVD(nb_dimensions)
            final_features = tsvd.fit_transform(flatten_features)
            n_explain = np.sum(np.cumsum(tsvd.explained_variance_ratio_) <= n_pca_features) + 1
            final_features = final_features[:, :n_explain]
            n_pca_features = final_features.shape[1]
        elif isinstance(n_pca_features, int):
            if flatten_features.shape[1] > n_pca_features:
                if projection_mode == "pca":
                    from sklearn.decomposition import PCA

                    tsvd = PCA(n_pca_features, whiten=True)
                elif projection_mode == "tsvd":
                    from sklearn.decomposition import TruncatedSVD

                    tsvd = TruncatedSVD(n_pca_features)

                final_features = tsvd.fit_transform(flatten_features)
            else:
                final_features = flatten_features
                tsvd = None

        if clusterer == "hdbscan":
            from hdbscan import HDBSCAN

            clust = HDBSCAN(**clusterer_kwargs, core_dist_n_jobs=1)
            clust.fit(final_features)
            possible_labels = clust.labels_
            is_split = np.setdiff1d(possible_labels, [-1]).size > 1
            del clust
        elif clusterer == "isocut5":
            min_cluster_size = clusterer_kwargs["min_cluster_size"]
            dipscore, cutpoint = isocut5(final_features[:, 0])
            possible_labels = np.zeros(final_features.shape[0])
            min_dip = clusterer_kwargs.get("min_dip", 1.5)
            if dipscore > min_dip:
                mask = final_features[:, 0] > cutpoint
                if np.sum(mask) > min_cluster_size and np.sum(~mask):
                    possible_labels[mask] = 1
                is_split = np.setdiff1d(possible_labels, [-1]).size > 1
            else:
                is_split = False
        else:
            raise ValueError(f"wrong clusterer {clusterer}. Possible options are 'hdbscan' or 'isocut5'.")

        DEBUG = False  # only for Sam or dirty hacking

        if debug_folder is not None or DEBUG:
            import matplotlib.pyplot as plt

            labels_set = np.setdiff1d(possible_labels, [-1])
            colors = plt.colormaps["tab10"].resampled(len(labels_set))
            colors = {k: colors(i) for i, k in enumerate(labels_set)}
            colors[-1] = "k"
            fig, axs = plt.subplots(nrows=4)

            flatten_wfs = aligned_wfs.swapaxes(1, 2).reshape(aligned_wfs.shape[0], -1)

            if final_features.shape[1] == 1:
                final_features = np.hstack((final_features, np.zeros_like(final_features)))

            sl = slice(None, None, 100)
            for k in np.unique(possible_labels):
                mask = possible_labels == k
                ax = axs[0]
                ax.scatter(final_features[:, 0][mask], final_features[:, 1][mask], s=5, color=colors[k])
                if k > -1:
                    centroid = final_features[:, :2][mask].mean(axis=0)
                    ax.text(centroid[0], centroid[1], f"Label {k}", fontsize=10, color="k")
                ax = axs[1]
                ax.plot(flatten_wfs[mask].T, color=colors[k], alpha=0.1)
                if k > -1:
                    ax.plot(np.median(flatten_wfs[mask].T, axis=1), color=colors[k], lw=2)
                ax.set_xlabel("PCA features")

                ax = axs[3]
                if n_pca_features == 1:
                    bins = np.linspace(final_features[:, 0].min(), final_features[:, 0].max(), 100)
                    ax.hist(final_features[mask, 0], bins, color=colors[k], alpha=0.1)
                else:
                    ax.plot(final_features[mask].T, color=colors[k], alpha=0.1)
                if k > -1 and n_pca_features > 1:
                    ax.plot(np.median(final_features[mask].T, axis=1), color=colors[k], lw=2)
                ax.set_xlabel("Projected PCA features")

            if tsvd is not None:
                ax = axs[2]
                sorted_components = np.argsort(tsvd.explained_variance_ratio_)[::-1]
                ax.plot(tsvd.explained_variance_ratio_[sorted_components], c="k")
                del tsvd

            ymin, ymax = ax.get_ylim()
            ax.plot([n_pca_features, n_pca_features], [ymin, ymax], "k--")

            axs[0].set_title(f"{clusterer} level={recursion_level}")
            if not DEBUG:
                fig.savefig(str(debug_folder) + ".png")
                plt.close(fig)
            else:
                plt.show()

        if not is_split:
            return is_split, None

        local_labels[kept] = possible_labels

        return is_split, local_labels


split_methods_list = [
    LocalFeatureClustering,
]
split_methods_dict = {e.name: e for e in split_methods_list}
