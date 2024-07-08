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

    from .isocut5 import isocut5

except:
    pass
from .tools import aggregate_sparse_features, FeaturesLoader, compute_template_from_sparse


DEBUG = False


def merge_clusters(
    peaks,
    peak_labels,
    recording,
    features_dict_or_folder,
    radius_um=70,
    method="waveforms_lda",
    method_kwargs={},
    **job_kwargs,
):
    """
    Merge cluster using differents methods.

    Parameters
    ----------
    peaks: numpy.ndarray 1d
        detected peaks (or a subset)
    peak_labels: numpy.ndarray 1d
        original label before merge
        peak_labels.size == peaks.size
    recording: Recording object
        A recording object
    features_dict_or_folder: dict or folder
        A dictionary of features precomputed with peak_pipeline or a folder containing npz file for features.
    method: str
        The method used
    method_kwargs: dict
        Option for the method.
    Returns
    -------
    merge_peak_labels: numpy.ndarray 1d
        New vectors label after merges.
    peak_shifts: numpy.ndarray 1d
        A vector of sample shift to be reverse applied on original sample_index on peak detection
        Negative shift means too early.
        Posituve shift means too late.
        So the correction must be applied like this externaly:
        final_peaks = peaks.copy()
        final_peaks['sample_index'] -= peak_shifts

    """

    job_kwargs = fix_job_kwargs(job_kwargs)

    features = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)
    sparse_wfs = features["sparse_wfs"]
    sparse_mask = features["sparse_mask"]

    labels_set, pair_mask, pair_shift, pair_values = find_merge_pairs(
        peaks,
        peak_labels,
        recording,
        features_dict_or_folder,
        sparse_wfs,
        sparse_mask,
        radius_um=radius_um,
        method=method,
        method_kwargs=method_kwargs,
        **job_kwargs,
    )

    if DEBUG:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.matshow(pair_values)

        pair_values[~pair_mask] = 20

        import hdbscan

        fig, ax = plt.subplots()
        clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=2, allow_single_cluster=True)
        clusterer.fit(pair_values)
        print(clusterer.labels_)
        clusterer.single_linkage_tree_.plot(cmap="viridis", colorbar=True)
        # ~ fig, ax = plt.subplots()
        # ~ clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
        # ~ edge_alpha=0.6,
        # ~ node_size=80,
        # ~ edge_linewidth=2)

        graph = clusterer.single_linkage_tree_.to_networkx()

        import scipy.cluster

        fig, ax = plt.subplots()
        scipy.cluster.hierarchy.dendrogram(clusterer.single_linkage_tree_.to_numpy(), ax=ax)

        import networkx as nx

        fig = plt.figure()
        nx.draw_networkx(graph)
        plt.show()

        plt.show()

    merges = agglomerate_pairs(labels_set, pair_mask, pair_values, connection_mode="partial")
    # merges = agglomerate_pairs(labels_set, pair_mask, pair_values, connection_mode="full")

    group_shifts = resolve_final_shifts(labels_set, merges, pair_mask, pair_shift)

    # apply final label and shift
    merge_peak_labels = peak_labels.copy()
    peak_shifts = np.zeros(peak_labels.size, dtype="int64")
    for merge, shifts in zip(merges, group_shifts):
        label0 = merge[0]
        mask = np.in1d(peak_labels, merge)
        merge_peak_labels[mask] = label0
        for l, label1 in enumerate(merge):
            if l == 0:
                # the first label is the reference (shift=0)
                continue
            peak_shifts[peak_labels == label1] = shifts[l]

    return merge_peak_labels, peak_shifts


def resolve_final_shifts(labels_set, merges, pair_mask, pair_shift):
    labels_set = list(labels_set)

    group_shifts = []
    for merge in merges:
        shifts = np.zeros(len(merge), dtype="int64")

        label_inds = [labels_set.index(label) for label in merge]

        label0 = merge[0]
        ind0 = label_inds[0]

        # First find relative shift to label0 (l=0) in the subgraph
        local_pair_mask = pair_mask[label_inds, :][:, label_inds]
        local_pair_shift = None
        G = None
        for l, label1 in enumerate(merge):
            if l == 0:
                # the first label is the reference (shift=0)
                continue
            ind1 = label_inds[l]
            if local_pair_mask[0, l]:
                # easy case the pair label0<>label1 was existing
                shift = pair_shift[ind0, ind1]
            else:
                # more complicated case need to find intermediate label and propagate the shift!!
                if G is None:
                    # the the graph only once and only if needed
                    G = nx.from_numpy_array(local_pair_mask | local_pair_mask.T)
                    local_pair_shift = pair_shift[label_inds, :][:, label_inds]
                    local_pair_shift += local_pair_shift.T

                shift_chain = nx.shortest_path(G, source=l, target=0)
                shift = 0
                for i in range(len(shift_chain) - 1):
                    shift += local_pair_shift[shift_chain[i + 1], shift_chain[i]]
            shifts[l] = shift

        group_shifts.append(shifts)

    return group_shifts


def agglomerate_pairs(labels_set, pair_mask, pair_values, connection_mode="full"):
    """
    Agglomerate merge pairs into final merge groups.

    The merges are ordered by label.

    """

    labels_set = np.array(labels_set)

    merges = []

    graph = nx.from_numpy_array(pair_mask | pair_mask.T)
    # put real nodes names for debugging
    maps = dict(zip(np.arange(labels_set.size), labels_set))
    graph = nx.relabel_nodes(graph, maps)

    groups = list(nx.connected_components(graph))
    for group in groups:
        if len(group) == 1:
            continue
        sub_graph = graph.subgraph(group)
        # print(group, sub_graph)
        cliques = list(nx.find_cliques(sub_graph))
        if len(cliques) == 1 and len(cliques[0]) == len(group):
            # the sub graph is full connected: no ambiguity
            # merges.append(labels_set[cliques[0]])
            merges.append(cliques[0])
        elif len(cliques) > 1:
            # the subgraph is not fully connected
            if connection_mode == "full":
                # node merge
                pass
            elif connection_mode == "partial":
                group = list(group)
                # merges.append(labels_set[group])
                merges.append(group)
            elif connection_mode == "clique":
                raise NotImplementedError
            else:
                raise ValueError

            if DEBUG:
                import matplotlib.pyplot as plt

                fig = plt.figure()
                nx.draw_networkx(sub_graph)
                plt.show()

    if DEBUG:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        nx.draw_networkx(graph)
        plt.show()

    # ensure ordered label
    merges = [np.sort(merge) for merge in merges]

    return merges


def find_merge_pairs(
    peaks,
    peak_labels,
    recording,
    features_dict_or_folder,
    sparse_wfs,
    sparse_mask,
    radius_um=70,
    method="project_distribution",
    method_kwargs={},
    **job_kwargs,
    # n_jobs=1,
    # mp_context="fork",
    # max_threads_per_process=1,
    # progress_bar=True,
):
    """
    Searh some possible merge 2 by 2.
    """
    job_kwargs = fix_job_kwargs(job_kwargs)

    # features_dict_or_folder = Path(features_dict_or_folder)

    # peaks = features_dict_or_folder['peaks']
    total_channels = recording.get_num_channels()

    # sparse_wfs = features['sparse_wfs']

    labels_set = np.setdiff1d(peak_labels, [-1]).tolist()
    n = len(labels_set)
    pair_mask = np.triu(np.ones((n, n), dtype="bool")) & ~np.eye(n, dtype="bool")
    pair_shift = np.zeros((n, n), dtype="int64")
    pair_values = np.zeros((n, n), dtype="float64")

    # compute template (no shift at this step)

    templates = compute_template_from_sparse(
        peaks, peak_labels, labels_set, sparse_wfs, sparse_mask, total_channels, peak_shifts=None
    )

    max_chans = np.argmax(np.max(np.abs(templates), axis=1), axis=1)

    channel_locs = recording.get_channel_locations()
    template_locs = channel_locs[max_chans, :]
    template_dist = scipy.spatial.distance.cdist(template_locs, template_locs, metric="euclidean")

    pair_mask = pair_mask & (template_dist <= radius_um)
    indices0, indices1 = np.nonzero(pair_mask)

    n_jobs = job_kwargs["n_jobs"]
    mp_context = job_kwargs.get("mp_context", None)
    max_threads_per_process = job_kwargs.get("max_threads_per_process", 1)
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
            templates,
            method,
            method_kwargs,
            max_threads_per_process,
        ),
    ) as pool:
        jobs = []
        for ind0, ind1 in zip(indices0, indices1):
            label0 = labels_set[ind0]
            label1 = labels_set[ind1]
            jobs.append(pool.submit(find_pair_function_wrapper, label0, label1))

        if progress_bar:
            iterator = tqdm(jobs, desc=f"find_merge_pairs with {method}", total=len(jobs))
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
    templates,
    method,
    method_kwargs,
    max_threads_per_process,
):
    global _ctx
    _ctx = {}

    _ctx["recording"] = recording
    _ctx["original_labels"] = original_labels
    _ctx["labels_set"] = labels_set
    _ctx["templates"] = templates
    _ctx["method"] = method
    _ctx["method_kwargs"] = method_kwargs
    _ctx["method_class"] = find_pair_method_dict[method]
    _ctx["max_threads_per_process"] = max_threads_per_process

    # if isinstance(features_dict_or_folder, dict):
    #     _ctx["features"] = features_dict_or_folder
    # else:
    #     _ctx["features"] = FeaturesLoader(features_dict_or_folder)

    _ctx["features"] = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)

    _ctx["peaks"] = _ctx["features"]["peaks"]


def find_pair_function_wrapper(label0, label1):
    global _ctx
    with threadpool_limits(limits=_ctx["max_threads_per_process"]):
        is_merge, label0, label1, shift, merge_value = _ctx["method_class"].merge(
            label0,
            label1,
            _ctx["labels_set"],
            _ctx["templates"],
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
        templates,
        original_labels,
        peaks,
        features,
        waveforms_sparse_mask=None,
        feature_name="sparse_tsvd",
        projection="centroid",
        criteria="diptest",
        threshold_diptest=0.5,
        threshold_percentile=80.0,
        threshold_overlap=0.4,
        min_cluster_size=50,
        num_shift=2,
    ):
        if num_shift > 0:
            assert feature_name == "sparse_wfs"
        sparse_wfs = features[feature_name]

        assert waveforms_sparse_mask is not None

        (inds0,) = np.nonzero(original_labels == label0)
        chans0 = np.unique(peaks["channel_index"][inds0])
        target_chans0 = np.flatnonzero(np.all(waveforms_sparse_mask[chans0, :], axis=0))

        (inds1,) = np.nonzero(original_labels == label1)
        chans1 = np.unique(peaks["channel_index"][inds1])
        target_chans1 = np.flatnonzero(np.all(waveforms_sparse_mask[chans1, :], axis=0))

        if inds0.size < min_cluster_size or inds1.size < min_cluster_size:
            is_merge = False
            merge_value = 0
            final_shift = 0
            return is_merge, label0, label1, final_shift, merge_value

        target_chans = np.intersect1d(target_chans0, target_chans1)

        inds = np.concatenate([inds0, inds1])
        labels = np.zeros(inds.size, dtype="int")
        labels[inds0.size :] = 1
        wfs, out = aggregate_sparse_features(peaks, inds, sparse_wfs, waveforms_sparse_mask, target_chans)
        wfs = wfs[~out]
        labels = labels[~out]

        cut = np.searchsorted(labels, 1)
        wfs0_ = wfs[:cut, :, :]
        wfs1_ = wfs[cut:, :, :]

        template0_ = np.mean(wfs0_, axis=0)
        template1_ = np.mean(wfs1_, axis=0)
        num_samples = template0_.shape[0]

        template0 = template0_[num_shift : num_samples - num_shift, :]

        wfs0 = wfs0_[:, num_shift : num_samples - num_shift, :]

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
        channel_shift = np.argmax(np.abs(template1_), axis=0) - np.argmax(np.abs(template0_), axis=0)
        mask = np.abs(channel_shift) <= num_shift
        channel_shift = channel_shift[mask]
        if channel_shift.size > 0:
            best_shift = int(np.round(np.mean(channel_shift))) + num_shift
        else:
            best_shift = num_shift

        wfs1 = wfs1_[:, best_shift : best_shift + template0.shape[0], :]
        template1 = template1_[best_shift : best_shift + template0.shape[0], :]

        if projection == "lda":
            wfs_0_1 = np.concatenate([wfs0, wfs1], axis=0)
            flat_wfs = wfs_0_1.reshape(wfs_0_1.shape[0], -1)
            feat = LinearDiscriminantAnalysis(n_components=1).fit_transform(flat_wfs, labels)
            feat = feat[:, 0]
            feat0 = feat[:cut]
            feat1 = feat[cut:]

        elif projection == "centroid":
            vector_0_1 = template1 - template0
            vector_0_1 /= np.sum(vector_0_1**2)
            feat0 = np.sum((wfs0 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            feat1 = np.sum((wfs1 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            # feat  = np.sum((wfs_0_1 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            feat = np.concatenate([feat0, feat1], axis=0)

        else:
            raise ValueError(f"bad projection {projection}")

        if criteria == "diptest":
            dipscore, cutpoint = isocut5(feat)
            is_merge = dipscore < threshold_diptest
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
            final_shift = best_shift - num_shift
        else:
            final_shift = 0

        if DEBUG:
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

            if criteria == "diptest":
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


class NormalizedTemplateDiff:
    """
    Compute the normalized (some kind of) template differences.
    And merge if below a threhold.
    Do this at several shift.

    """

    name = "normalized_template_diff"

    @staticmethod
    def merge(
        label0,
        label1,
        labels_set,
        templates,
        original_labels,
        peaks,
        features,
        waveforms_sparse_mask=None,
        threshold_diff=1.5,
        min_cluster_size=50,
        num_shift=5,
    ):
        assert waveforms_sparse_mask is not None

        (inds0,) = np.nonzero(original_labels == label0)
        chans0 = np.unique(peaks["channel_index"][inds0])
        target_chans0 = np.flatnonzero(np.all(waveforms_sparse_mask[chans0, :], axis=0))

        (inds1,) = np.nonzero(original_labels == label1)
        chans1 = np.unique(peaks["channel_index"][inds1])
        target_chans1 = np.flatnonzero(np.all(waveforms_sparse_mask[chans1, :], axis=0))

        # if inds0.size < min_cluster_size or inds1.size < min_cluster_size:
        #     is_merge = False
        #     merge_value = 0
        #     final_shift = 0
        #     return is_merge, label0, label1, final_shift, merge_value

        target_chans = np.intersect1d(target_chans0, target_chans1)
        union_chans = np.union1d(target_chans0, target_chans1)

        ind0 = list(labels_set).index(label0)
        template0 = templates[ind0][:, target_chans]

        ind1 = list(labels_set).index(label1)
        template1 = templates[ind1][:, target_chans]

        num_samples = template0.shape[0]
        # norm = np.mean(np.abs(template0)) + np.mean(np.abs(template1))
        norm = np.mean(np.abs(template0) + np.abs(template1))

        # norm_per_channel = np.max(np.abs(template0) + np.abs(template1), axis=0) / 2.
        norm_per_channel = (np.max(np.abs(template0), axis=0) + np.max(np.abs(template1), axis=0)) * 0.5
        # norm_per_channel = np.max(np.abs(template0)) + np.max(np.abs(template1)) / 2.
        # print(norm_per_channel)

        all_shift_diff = []
        # all_shift_diff_by_channel = []
        for shift in range(-num_shift, num_shift + 1):
            temp0 = template0[num_shift : num_samples - num_shift, :]
            temp1 = template1[num_shift + shift : num_samples - num_shift + shift, :]
            # d = np.mean(np.abs(temp0 - temp1)) / (norm)
            # d = np.max(np.abs(temp0 - temp1)) / (norm)
            diff_per_channel = np.abs(temp0 - temp1) / norm

            diff_max = np.max(diff_per_channel, axis=0)

            # diff = np.max(diff_per_channel)
            diff = np.average(diff_max, weights=norm_per_channel)
            # diff = np.average(diff_max)
            all_shift_diff.append(diff)
            # diff_by_channel = np.mean(np.abs(temp0 - temp1), axis=0) / (norm)
            # all_shift_diff_by_channel.append(diff_by_channel)
            # d = np.mean(diff_by_channel)
            # all_shift_diff.append(d)
        normed_diff = np.min(all_shift_diff)

        is_merge = normed_diff < threshold_diff

        if is_merge:
            merge_value = normed_diff
            final_shift = np.argmin(all_shift_diff) - num_shift

            # diff_by_channel = all_shift_diff_by_channel[np.argmin(all_shift_diff)]
        else:
            final_shift = 0
            merge_value = np.nan

        # print('merge_value', merge_value, 'final_shift', final_shift, 'is_merge', is_merge)

        DEBUG = False
        # DEBUG = True
        # if DEBUG and ( 0. < normed_diff < .4):
        # if 0.5 < normed_diff < 4:
        if DEBUG and is_merge:
            # if DEBUG:

            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(nrows=3)

            temp0 = template0[num_shift : num_samples - num_shift, :]
            temp1 = template1[num_shift + final_shift : num_samples - num_shift + final_shift, :]

            diff_per_channel = np.abs(temp0 - temp1) / norm
            diff = np.max(diff_per_channel)

            m0 = temp0.T.flatten()
            m1 = temp1.T.flatten()

            ax = axs[0]
            ax.plot(m0, color="C0", label=f"{label0} {inds0.size}")
            ax.plot(m1, color="C1", label=f"{label1} {inds1.size}")

            ax.set_title(
                f"union{union_chans.size} intersect{target_chans.size} \n {normed_diff:.3f} {final_shift} {is_merge}"
            )
            ax.legend()

            ax = axs[1]

            # ~ temp0 = template0[num_shift : num_samples - num_shift, :]
            # ~ temp1 = template1[num_shift + shift : num_samples - num_shift + shift, :]
            ax.plot(np.abs(m0 - m1))
            # ax.axhline(norm, ls='--', color='k')
            ax = axs[2]
            ax.plot(diff_per_channel.T.flatten())
            ax.axhline(threshold_diff, ls="--")
            ax.axhline(normed_diff)

            # ax.axhline(normed_diff, ls='-', color='b')
            # ax.plot(norm, ls='--')
            # ax.plot(diff_by_channel)

            # ax.plot(np.abs(m0) + np.abs(m1))

            # ax.plot(np.abs(m0 - m1) / (np.abs(m0) + np.abs(m1)))

            # ax.set_title(f"{norm=:.3f}")

            plt.show()

        return is_merge, label0, label1, final_shift, merge_value


find_pair_method_list = [
    ProjectDistribution,
    NormalizedTemplateDiff,
]
find_pair_method_dict = {e.name: e for e in find_pair_method_list}
