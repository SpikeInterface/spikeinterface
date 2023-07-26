from pathlib import Path
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

import scipy.spatial
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from hdbscan import HDBSCAN

import numpy as np


from spikeinterface.core.job_tools import get_poolexecutor, fix_job_kwargs


from .isocut5 import isocut5

from .tools import aggregate_sparse_features, FeaturesLoader, compute_template_from_sparse


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
    peak_labels: numpy.ndarray 1d
        New vectors label after merges.
    """

    job_kwargs = fix_job_kwargs(job_kwargs)
    print(job_kwargs)

    labels_set = np.setdiff1d(peak_labels, [-1])

    features = FeaturesLoader.from_dict_or_folder(features_dict_or_folder)
    sparse_wfs = features["sparse_wfs"]
    sparse_mask = features["sparse_mask"]

    pair_mask, pair_shift = find_merge_pairs(
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

    merges = find_connected_pairs(pair_mask, labels_set, connection_mode="partial")

    peak_labels = peak_labels.copy()

    for merge in merges:
        mask = np.in1d(peak_labels, merge)
        peak_labels[mask] = min(merge)

    return peak_labels


def find_connected_pairs(pair_mask, labels_set, connection_mode="full"):
    import networkx as nx

    labels_set = np.array(labels_set)

    merges = []

    graph = nx.from_numpy_matrix(pair_mask | pair_mask.T)
    groups = list(nx.connected_components(graph))
    for group in groups:
        if len(group) == 1:
            continue
        sub_graph = graph.subgraph(group)
        # print(group, sub_graph)
        cliques = list(nx.find_cliques(sub_graph))
        if len(cliques) == 1 and len(cliques[0]) == len(group):
            # the sub graph is full connected: no ambiguity
            merges.append(labels_set[cliques[0]])
        elif len(cliques) > 1:
            # the subgraph is not fully connected
            if connection_mode == "full":
                # node merge
                pass
            elif connection_mode == "partial":
                group = list(group)
                merges.append(labels_set[group])
            elif connection_mode == "clique":
                raise NotImplementedError
            else:
                raise ValueError

            # DEBUG = True
            DEBUG = False
            if DEBUG:
                import matplotlib.pyplot as plt

                fig = plt.figure()
                nx.draw_networkx(sub_graph)
                plt.show()

    # DEBUG = True
    DEBUG = False
    if DEBUG:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        nx.draw_networkx(graph)
        plt.show()

    return merges


def find_merge_pairs(
    peaks,
    peak_labels,
    recording,
    features_dict_or_folder,
    sparse_wfs,
    sparse_mask,
    radius_um=70,
    method="waveforms_lda",
    method_kwargs={},
    n_jobs=1,
    mp_context="fork",
    max_threads_per_process=1,
    progress_bar=True,
):
    """
    Try some merges on clusters in parralel


    """
    # features_dict_or_folder = Path(features_dict_or_folder)

    # peaks = features_dict_or_folder['peaks']
    total_channels = recording.get_num_channels()

    # sparse_wfs = features['sparse_wfs']

    labels_set = np.setdiff1d(peak_labels, [-1]).tolist()
    n = len(labels_set)
    pair_mask = np.triu(np.ones((n, n), dtype="bool")) & ~np.eye(n, dtype="bool")
    pair_shift = np.zeros((n, n), dtype="int64")

    # compute template

    templates = compute_template_from_sparse(peaks, peak_labels, labels_set, sparse_wfs, sparse_mask, total_channels)

    max_chans = np.argmax(np.max(np.abs(templates), axis=1), axis=1)

    channel_locs = recording.get_channel_locations()
    template_locs = channel_locs[max_chans, :]
    template_dist = scipy.spatial.distance.cdist(template_locs, template_locs, metric="euclidean")

    pair_mask = pair_mask & (template_dist < radius_um)
    indices0, indices1 = np.nonzero(pair_mask)

    Executor = get_poolexecutor(n_jobs)

    with Executor(
        max_workers=n_jobs,
        initializer=find_pair_worker_init,
        mp_context=get_context(mp_context),
        initargs=(recording, features_dict_or_folder, peak_labels, method, method_kwargs, max_threads_per_process),
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
            is_merge, label0, label1, shift = res.result()
            ind0 = labels_set.index(label0)
            ind1 = labels_set.index(label1)

            pair_mask[ind0, ind1] = is_merge
            if is_merge:
                pair_shift[ind0, ind1] = shift

    pair_mask = pair_mask & (template_dist < radius_um)
    indices0, indices1 = np.nonzero(pair_mask)

    return pair_mask, pair_shift


def find_pair_worker_init(
    recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_process
):
    global _ctx
    _ctx = {}

    _ctx["recording"] = recording
    _ctx["original_labels"] = original_labels
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
        is_merge, label0, label1, shift = _ctx["method_class"].merge(
            label0, label1, _ctx["original_labels"], _ctx["peaks"], _ctx["features"], **_ctx["method_kwargs"]
        )
    return is_merge, label0, label1, shift


class WaveformsLda:
    name = "waveforms_lda"

    @staticmethod
    def merge(
        label0,
        label1,
        original_labels,
        peaks,
        features,
        waveforms_sparse_mask=None,
        feature_name="sparse_tsvd",
        projection="centroid",
        criteria="diptest",
        threshold_diptest=0.5,
        threshold_percentile=80.0,
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

        values = []
        for shift in range(num_shift * 2 + 1):
            template1 = template1_[shift : shift + template0.shape[0], :]

            norm = np.linalg.norm(template0.flatten()) * np.linalg.norm(template1.flatten())
            value = np.sum(template0.flatten() * template1.flatten()) / norm
            values.append(value)

        best_shift = np.argmax(values)

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
        elif criteria == "percentile":
            l0 = np.percentile(feat0, threshold_percentile)
            l1 = np.percentile(feat1, 100.0 - threshold_percentile)
            is_merge = l0 >= l1
        else:
            raise ValueError(f"bad criteria {criteria}")

        if is_merge:
            final_shift = best_shift - num_shift
        else:
            final_shift = 0

        # DEBUG = True
        DEBUG = False
        if DEBUG and is_merge:
            import matplotlib.pyplot as plt

            flatten_wfs0 = wfs0.swapaxes(1, 2).reshape(wfs0.shape[0], -1)
            flatten_wfs1 = wfs1.swapaxes(1, 2).reshape(wfs1.shape[0], -1)

            fig, axs = plt.subplots(ncols=2)
            ax = axs[0]
            ax.plot(flatten_wfs0.T, color="C0", alpha=0.01)
            ax.plot(flatten_wfs1.T, color="C1", alpha=0.01)
            m0 = np.mean(flatten_wfs0, axis=0)
            m1 = np.mean(flatten_wfs1, axis=0)
            ax.plot(m0, color="C0", alpha=1, lw=4)
            ax.plot(m1, color="C1", alpha=1, lw=4)

            bins = np.linspace(np.percentile(feat, 1), np.percentile(feat, 99), 100)

            count0, _ = np.histogram(feat0, bins=bins)
            count1, _ = np.histogram(feat1, bins=bins)

            ax = axs[1]
            ax.plot(bins[:-1], count0, color="C0")
            ax.plot(bins[:-1], count1, color="C1")

            ax.set_title(f"{dipscore}")

        return is_merge, label0, label1, final_shift


find_pair_method_list = [
    WaveformsLda,
]
find_pair_method_dict = {e.name: e for e in find_pair_method_list}
