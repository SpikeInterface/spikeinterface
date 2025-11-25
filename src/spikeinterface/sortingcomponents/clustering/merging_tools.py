from __future__ import annotations

from multiprocessing import get_context
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


import numpy as np

from spikeinterface.core.job_tools import get_poolexecutor, fix_job_kwargs

try:
    import numba
    import networkx as nx
    import scipy.spatial
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from .isosplit_isocut import isocut

except:
    pass
from .tools import aggregate_sparse_features, FeaturesLoader


DEBUG = False


def merge_peak_labels_from_features(
    peaks,
    peak_labels,
    unit_ids,
    templates_array,
    template_sparse_mask,
    recording,
    features_dict_or_folder,
    radius_um=70.0,
    method="project_distribution",
    method_kwargs={},
    job_kwargs=None,
):
    """
    Merge cluster from all features with distribution pair by pair.
    Support eventually multi method.
    """

    job_kwargs = fix_job_kwargs(job_kwargs)

    features = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)
    # sparse_wfs = features["sparse_wfs"]
    # sparse_mask = features["sparse_mask"]

    labels_set, pair_mask, pair_shift, pair_values = find_merge_pairs_from_features(
        peaks,
        peak_labels,
        unit_ids,
        templates_array,
        template_sparse_mask,
        recording,
        features_dict_or_folder,
        # sparse_wfs,
        # sparse_mask,
        radius_um=radius_um,
        method=method,
        method_kwargs=method_kwargs,
        job_kwargs=job_kwargs,
    )

    clean_labels, merge_template_array, merge_sparsity_mask, new_unit_ids = (
        _apply_pair_mask_on_labels_and_recompute_templates(
            pair_mask, peak_labels, unit_ids, templates_array, template_sparse_mask
        )
    )

    return clean_labels, merge_template_array, merge_sparsity_mask, new_unit_ids


def find_merge_pairs_from_features(
    peaks,
    peak_labels,
    unit_ids,
    templates_array,
    template_sparse_mask,
    recording,
    features_dict_or_folder,
    # sparse_wfs,
    # sparse_mask,
    radius_um=70,
    method="project_distribution",
    method_kwargs={},
    job_kwargs=None,
    # n_jobs=1,
    # mp_context="fork",
    # max_threads_per_worker=1,
    # progress_bar=True,
):
    """
    Search some possible merge 2 by 2.
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    # features_dict_or_folder = Path(features_dict_or_folder)
    # features = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)

    # peaks = features_dict_or_folder['peaks']
    # total_channels = recording.get_num_channels()

    # sparse_wfs = features['sparse_wfs']

    n = len(unit_ids)
    pair_mask = np.triu(np.ones((n, n), dtype="bool")) & ~np.eye(n, dtype="bool")
    pair_shift = np.zeros((n, n), dtype="int64")
    pair_values = np.zeros((n, n), dtype="float64")

    # compute template (no shift at this step)

    # templates = compute_template_from_sparse(
    #     peaks, peak_labels, labels_set, sparse_wfs, sparse_mask, total_channels, peak_shifts=None
    # )

    # peaks_svd = features['peaks_svd']
    # sparse_mask = features['sparse_mask']
    # ms_before = features['ms_before']
    # ms_after = features['ms_after']
    # svd_model = features['svd_model']

    # templates, final_sparsity_mask = get_templates_from_peaks_and_svd(
    #     recording, peaks, peak_labels, ms_before, ms_after, svd_model, peaks_svd, sparse_mask, operator="average",
    # )
    # dense_templates_array = templates.templates_array

    labels_set = unit_ids.tolist()

    max_chans = np.argmax(np.max(np.abs(templates_array), axis=1), axis=1)

    channel_locs = recording.get_channel_locations()
    template_locs = channel_locs[max_chans, :]
    template_dist = scipy.spatial.distance.cdist(template_locs, template_locs, metric="euclidean")

    # print("template_locs", template_locs.shape, template_locs)
    # print("template_locs", np.unique(template_locs[:, 1]).shape)
    # print("radius_um", radius_um)

    pair_mask = pair_mask & (template_dist <= radius_um)
    indices0, indices1 = np.nonzero(pair_mask)

    n_jobs = job_kwargs["n_jobs"]
    mp_context = job_kwargs.get("mp_context", None)
    max_threads_per_worker = job_kwargs.get("max_threads_per_worker", 1)
    progress_bar = job_kwargs["progress_bar"]

    Executor = get_poolexecutor(n_jobs)

    with Executor(
        max_workers=n_jobs,
        initializer=find_pair_worker_init,
        mp_context=get_context(mp_context),
        initargs=(
            recording,
            features_dict_or_folder,
            peak_labels,
            labels_set,
            templates_array,
            method,
            method_kwargs,
            max_threads_per_worker,
        ),
    ) as pool:
        jobs = []
        for ind0, ind1 in zip(indices0, indices1):
            label0 = labels_set[ind0]
            label1 = labels_set[ind1]
            jobs.append(pool.submit(find_pair_function_wrapper, label0, label1))

        if progress_bar:
            iterator = tqdm(jobs, desc=f"find_merge_pairs_from_features with {method}", total=len(jobs))
        else:
            iterator = jobs

        for res in iterator:
            is_merge, label0, label1, shift, merge_value = res.result()
            ind0 = labels_set.index(label0)
            ind1 = labels_set.index(label1)

            pair_mask[ind0, ind1] = is_merge
            if is_merge:
                pair_shift[ind0, ind1] = shift
                pair_values[ind0, ind1] = merge_value

    pair_mask = pair_mask & (template_dist <= radius_um)
    indices0, indices1 = np.nonzero(pair_mask)

    return labels_set, pair_mask, pair_shift, pair_values


def find_pair_worker_init(
    recording,
    features_dict_or_folder,
    original_labels,
    labels_set,
    templates_array,
    method,
    method_kwargs,
    max_threads_per_worker,
):
    global _ctx
    _ctx = {}

    _ctx["recording"] = recording
    _ctx["original_labels"] = original_labels
    _ctx["labels_set"] = labels_set
    _ctx["templates_array"] = templates_array
    _ctx["method"] = method
    _ctx["method_kwargs"] = method_kwargs
    _ctx["method_class"] = find_pair_method_dict[method]
    _ctx["max_threads_per_worker"] = max_threads_per_worker

    # if isinstance(features_dict_or_folder, dict):
    #     _ctx["features"] = features_dict_or_folder
    # else:
    #     _ctx["features"] = FeaturesLoader(features_dict_or_folder)

    _ctx["features"] = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)

    _ctx["peaks"] = _ctx["features"]["peaks"]


def find_pair_function_wrapper(label0, label1):
    global _ctx
    with threadpool_limits(limits=_ctx["max_threads_per_worker"]):
        is_merge, label0, label1, shift, merge_value = _ctx["method_class"].merge(
            label0,
            label1,
            _ctx["labels_set"],
            _ctx["templates_array"],
            _ctx["original_labels"],
            _ctx["peaks"],
            _ctx["features"],
            **_ctx["method_kwargs"],
        )

    return is_merge, label0, label1, shift, merge_value


class ProjectDistribution:
    """
    This method is a refactorized mix  between:
       * old tridesclous code
       * some ideas by Charlie Windolf in spikespvae

    The idea is :
      * project the waveform (or features) samples on a 1d axis (using  LDA for instance).
      * check that it is the same or not distribution (diptest, distrib_overlap, ...)
    """

    name = "project_distribution"

    @staticmethod
    def merge(
        label0,
        label1,
        labels_set,
        templates_array,
        original_labels,
        peaks,
        features,
        waveforms_sparse_mask=None,
        feature_name="sparse_tsvd",
        projection="centroid",
        criteria="isocut",
        isocut_threshold=2.0,
        threshold_percentile=80.0,
        threshold_overlap=0.4,
        min_cluster_size=50,
        # num_shift=2,
        n_pca_features=3,
        seed=None,
        projection_mode="tsvd",
        minimum_overlap_ratio=0.75,
    ):
        # if num_shift > 0:
        #     assert feature_name == "sparse_wfs"

        sparse_wfs = features[feature_name]
        # sparse_mask = features["sparse_mask"]
        sparse_mask = waveforms_sparse_mask

        assert sparse_mask is not None

        (inds0,) = np.nonzero(original_labels == label0)
        chans0 = np.unique(peaks["channel_index"][inds0])
        target_chans0 = np.flatnonzero(np.all(sparse_mask[chans0, :], axis=0))

        (inds1,) = np.nonzero(original_labels == label1)
        chans1 = np.unique(peaks["channel_index"][inds1])
        target_chans1 = np.flatnonzero(np.all(sparse_mask[chans1, :], axis=0))

        if inds0.size < min_cluster_size or inds1.size < min_cluster_size:
            is_merge = False
            merge_value = 0
            final_shift = 0
            return is_merge, label0, label1, final_shift, merge_value

        target_intersect_chans = np.intersect1d(target_chans0, target_chans1)
        target_union_chans = np.union1d(target_chans0, target_chans1)

        if (len(target_intersect_chans) / len(target_union_chans)) < minimum_overlap_ratio:
            is_merge = False
            merge_value = 0
            final_shift = 0
            return is_merge, label0, label1, final_shift, merge_value

        inds = np.concatenate([inds0, inds1])
        labels = np.zeros(inds.size, dtype="int")
        labels[inds0.size :] = 1
        wfs, out = aggregate_sparse_features(peaks, inds, sparse_wfs, sparse_mask, target_intersect_chans)
        wfs = wfs[~out]
        labels = labels[~out]

        cut = np.searchsorted(labels, 1)
        wfs0 = wfs[:cut, :, :]
        wfs1 = wfs[cut:, :, :]

        # num_samples = template0.shape[0]

        # template0 = template0_[num_shift : num_samples - num_shift, :]

        # wfs0 = wfs0_[:, num_shift : num_samples - num_shift, :]

        # best shift strategy 1 = max cosine
        # values = []
        # for shift in range(num_shift * 2 + 1):
        #     template1 = template1_[shift : shift + template0.shape[0], :]
        #     norm = np.linalg.norm(template0.flatten()) * np.linalg.norm(template1.flatten())
        #     value = np.sum(template0.flatten() * template1.flatten()) / norm
        #     values.append(value)
        # best_shift = np.argmax(values)

        # best shift strategy 2 = min dist**2
        # values = []
        # for shift in range(num_shift * 2 + 1):
        #     template1 = template1_[shift : shift + template0.shape[0], :]
        #     value = np.sum((template1 - template0)**2)
        #     values.append(value)
        # best_shift = np.argmin(values)

        # best shift strategy 3 : average delta argmin between channels
        # channel_shift = np.argmax(np.abs(template1_), axis=0) - np.argmax(np.abs(template0_), axis=0)
        # mask = np.abs(channel_shift) <= num_shift
        # channel_shift = channel_shift[mask]
        # if channel_shift.size > 0:
        #     best_shift = int(np.round(np.mean(channel_shift))) + num_shift
        # else:
        #     best_shift = num_shift

        # wfs1 = wfs1_[:, best_shift : best_shift + template0.shape[0], :]
        # template1 = template1_[best_shift : best_shift + template0.shape[0], :]

        feat0 = wfs0.reshape(wfs0.shape[0], -1)
        feat1 = wfs1.reshape(wfs1.shape[0], -1)
        feat = np.concatenate([feat0, feat1], axis=0)

        use_svd = True

        if use_svd:
            from sklearn.decomposition import TruncatedSVD

            n_pca_features = 3
            tsvd = TruncatedSVD(n_pca_features, random_state=seed)
            feat = tsvd.fit_transform(feat)

        if isinstance(n_pca_features, float):
            assert 0 < n_pca_features < 1, "n_components should be in ]0, 1["
            nb_dimensions = min(feat.shape[0], feat.shape[1])
            if projection_mode == "pca":
                from sklearn.decomposition import PCA

                tsvd = PCA(nb_dimensions, whiten=True)
            elif projection_mode == "tsvd":
                from sklearn.decomposition import TruncatedSVD

                tsvd = TruncatedSVD(nb_dimensions, random_state=seed)
            feat = tsvd.fit_transform(feat)
            n_explain = np.sum(np.cumsum(tsvd.explained_variance_ratio_) <= n_pca_features) + 1
            feat = feat[:, :n_explain]
            n_pca_features = feat.shape[1]
        elif isinstance(n_pca_features, int):
            if feat.shape[1] > n_pca_features:
                if projection_mode == "pca":
                    from sklearn.decomposition import PCA

                    tsvd = PCA(n_pca_features, whiten=True)

                elif projection_mode == "tsvd":
                    from sklearn.decomposition import TruncatedSVD

                    tsvd = TruncatedSVD(n_pca_features, random_state=seed)

                feat = tsvd.fit_transform(feat)

            else:
                feat = feat
                tsvd = None

        # else:
        #     feat = feat

        feat0 = feat[:cut]
        feat1 = feat[cut:]

        if projection == "lda":
            # wfs_0_1 = np.concatenate([wfs0, wfs1], axis=0)
            # flat_wfs = wfs_0_1.reshape(wfs_0_1.shape[0], -1)
            feat = LinearDiscriminantAnalysis(n_components=1).fit_transform(feat, labels)
            feat = feat[:, 0]

        elif projection == "centroid":
            # vector_0_1 = template1 - template0
            # vector_0_1 /= np.sum(vector_0_1**2)
            # feat0 = np.sum((wfs0 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            # feat1 = np.sum((wfs1 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            # # feat  = np.sum((wfs_0_1 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            # feat = np.concatenate([feat0, feat1], axis=0)

            # this is flatten
            template0 = np.mean(feat0, axis=0)
            template1 = np.mean(feat1, axis=0)
            vector_0_1 = template1 - template0
            vector_0_1 /= np.sum(vector_0_1**2)
            feat = np.sum((feat - template0[np.newaxis, :]) * vector_0_1[np.newaxis, :], axis=1)

        else:
            raise ValueError(f"bad projection {projection}")

        feat0 = feat[:cut]
        feat1 = feat[cut:]

        if criteria == "isocut":
            dipscore, cutpoint = isocut(feat)
            is_merge = dipscore < isocut_threshold
            merge_value = dipscore
        elif criteria == "percentile":
            l0 = np.percentile(feat0, threshold_percentile)
            l1 = np.percentile(feat1, 100.0 - threshold_percentile)
            is_merge = l0 >= l1
            merge_value = l0 - l1
        elif criteria == "distrib_overlap":

            lim0 = min(np.min(feat0), np.min(feat1))
            lim1 = max(np.max(feat0), np.max(feat1))
            bin_size = (lim1 - lim0) / 200.0
            bins = np.arange(lim0, lim1, bin_size)

            pdf0, _ = np.histogram(feat0, bins=bins, density=True)
            pdf1, _ = np.histogram(feat1, bins=bins, density=True)
            pdf0 *= bin_size
            pdf1 *= bin_size
            overlap = np.sum(np.minimum(pdf0, pdf1))

            is_merge = overlap >= threshold_overlap

            merge_value = 1 - overlap

        else:
            raise ValueError(f"bad criteria {criteria}")

        if is_merge:
            # final_shift = best_shift - num_shift
            final_shift = 0
        else:
            final_shift = 0

        if DEBUG:
            # if dipscore < 4:
            import matplotlib.pyplot as plt

            flatten_wfs0 = wfs0.swapaxes(1, 2).reshape(wfs0.shape[0], -1)
            flatten_wfs1 = wfs1.swapaxes(1, 2).reshape(wfs1.shape[0], -1)

            fig, axs = plt.subplots(ncols=2)
            ax = axs[0]
            ax.plot(flatten_wfs0.T, color="C0", alpha=0.01)
            ax.plot(flatten_wfs1.T, color="C1", alpha=0.01)
            m0 = np.mean(flatten_wfs0, axis=0)
            m1 = np.mean(flatten_wfs1, axis=0)
            ax.plot(m0, color="C0", alpha=1, lw=4, label=f"{label0} {inds0.size}")
            ax.plot(m1, color="C1", alpha=1, lw=4, label=f"{label1} {inds1.size}")

            ax.legend()

            bins = np.linspace(np.percentile(feat, 1), np.percentile(feat, 99), 100)
            bin_size = bins[1] - bins[0]
            count0, _ = np.histogram(feat0, bins=bins, density=True)
            count1, _ = np.histogram(feat1, bins=bins, density=True)
            pdf0 = count0 * bin_size
            pdf1 = count1 * bin_size

            ax = axs[1]
            ax.plot(bins[:-1], pdf0, color="C0")
            ax.plot(bins[:-1], pdf1, color="C1")

            if criteria == "isocut":
                ax.set_title(f"{dipscore:.4f} {is_merge}")
            elif criteria == "percentile":
                ax.set_title(f"{l0:.4f} {l1:.4f} {is_merge}")
                ax.axvline(l0, color="C0")
                ax.axvline(l1, color="C1")
            elif criteria == "distrib_overlap":
                print(
                    lim0,
                    lim1,
                )
                ax.set_title(f"{overlap:.4f} {is_merge}")
                ax.plot(bins[:-1], np.minimum(pdf0, pdf1), ls="--", color="k")

            plt.show()

        return is_merge, label0, label1, final_shift, merge_value


find_pair_method_list = [
    ProjectDistribution,
]
find_pair_method_dict = {e.name: e for e in find_pair_method_list}


def merge_peak_labels_from_templates(
    peaks,
    peak_labels,
    unit_ids,
    templates_array,
    template_sparse_mask,
    similarity_metric="l1",
    similarity_thresh=0.8,
    num_shifts=3,
    use_lags=False,
):
    """
    Low level function used in sorting components for merging templates based on similarity metrics.

    This is mostly used in clustering method to clean possible oversplits.

    templates_array is dense (num_templates, num_total_channel) but have a template_sparse_mask compagion
    """
    assert len(unit_ids) == templates_array.shape[0]

    from spikeinterface.postprocessing.template_similarity import compute_similarity_with_templates_array

    similarity, lags = compute_similarity_with_templates_array(
        templates_array,
        templates_array,
        method=similarity_metric,
        num_shifts=num_shifts,
        support="union",
        sparsity=template_sparse_mask,
        other_sparsity=template_sparse_mask,
    )

    pair_mask = similarity > similarity_thresh

    if not use_lags:
        lags = None

    clean_labels, merge_template_array, merge_sparsity_mask, new_unit_ids = (
        _apply_pair_mask_on_labels_and_recompute_templates(
            pair_mask, peak_labels, unit_ids, templates_array, template_sparse_mask, lags
        )
    )

    return clean_labels, merge_template_array, merge_sparsity_mask, new_unit_ids


def _apply_pair_mask_on_labels_and_recompute_templates(
    pair_mask, peak_labels, unit_ids, templates_array, template_sparse_mask, lags=None
):
    """
    Resolve pairs graph.
    Apply to new labels.
    Recompute templates.
    """

    from scipy.sparse.csgraph import connected_components

    keep_template = np.ones(templates_array.shape[0], dtype="bool")
    clean_labels = peak_labels.copy()
    n_components, group_labels = connected_components(pair_mask, directed=False, return_labels=True)

    # print("merges", templates_array.shape[0], "to", n_components)

    merge_template_array = templates_array.copy()
    merge_sparsity_mask = template_sparse_mask.copy()
    new_unit_ids = np.zeros(n_components, dtype=unit_ids.dtype)

    for c in range(n_components):
        merge_group = np.flatnonzero(group_labels == c)
        g0 = merge_group[0]
        new_unit_ids[c] = unit_ids[g0]
        if len(merge_group) > 1:
            weights = np.zeros(len(merge_group), dtype=np.float32)

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # for i, l in enumerate(merge_group):
            #     temp_flat = merge_template_array[l, :, :].T.flatten()
            #     ax.plot(temp_flat)
            # sim = similarity[merge_group[0], merge_group[1]]
            # ax.set_title(f"{sim} {similarity_thresh}")

            for i, l in enumerate(merge_group):
                label = unit_ids[l]
                weights[i] = np.sum(peak_labels == label)
                if i > 0:
                    clean_labels[peak_labels == label] = unit_ids[g0]
                    keep_template[l] = False
            weights /= weights.sum()

            if lags is None:
                merge_template_array[g0, :, :] = np.sum(
                    merge_template_array[merge_group, :, :] * weights[:, np.newaxis, np.newaxis], axis=0
                )
            else:
                # with shifts
                accumulated_template = np.zeros_like(merge_template_array[g0, :, :])
                for i, l in enumerate(merge_group):
                    shift = -lags[g0, l]
                    if shift > 0:
                        # template is shifted to right
                        temp = np.zeros_like(accumulated_template)
                        temp[shift:, :] = merge_template_array[l, :-shift, :]
                    elif shift < 0:
                        # template is shifted to left
                        temp = np.zeros_like(accumulated_template)
                        temp[:shift, :] = merge_template_array[l, -shift:, :]
                    else:
                        temp = merge_template_array[l, :, :]

                    accumulated_template += temp * weights[i]

                merge_template_array[g0, :, :] = accumulated_template
            merge_sparsity_mask[g0, :] = np.all(template_sparse_mask[merge_group, :], axis=0)

    merge_template_array = merge_template_array[keep_template, :, :]
    merge_sparsity_mask = merge_sparsity_mask[keep_template, :]

    return clean_labels, merge_template_array, merge_sparsity_mask, new_unit_ids
