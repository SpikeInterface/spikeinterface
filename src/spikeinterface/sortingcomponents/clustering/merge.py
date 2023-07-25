
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


from spikeinterface.core.job_tools import get_poolexecutor


from .isocut5 import isocut5

from .tools import aggregate_sparse_features, FeaturesLoader, compute_template_from_sparse

def merge_clusters():
    raise NotImplementedError



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

    original_labels = peak_labels
    peak_labels = peak_labels.copy()

    
    labels_set = np.setdiff1d(peak_labels, [-1]).tolist()
    n = len(labels_set)
    print('n', n)
    pair_mask = np.triu(np.ones((n, n), dtype="bool")) & ~np.eye(n, dtype="bool")
    pair_shift = np.zeros((n, n), dtype="int64")


    # compute template

    templates = compute_template_from_sparse(peaks, peak_labels, labels_set, sparse_wfs, sparse_mask, total_channels)

    max_chans = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    
    channel_locs = recording.get_channel_locations()
    template_locs = channel_locs[max_chans, :]
    # print(template_locs)

    template_dist = scipy.spatial.distance.cdist(template_locs, template_locs, metric="euclidean")
    # print(template_dist)

    
    # ind0, ind1 = np.nonzero(pair_mask)
    # print(ind0.size)
    pair_mask = pair_mask & (template_dist < radius_um)
    indices0, indices1 = np.nonzero(pair_mask)
    # print(indices0.size)


    Executor = get_poolexecutor(n_jobs)

    with Executor(
        max_workers=n_jobs,
        initializer=merge_worker_init,
        mp_context=get_context(mp_context),
        initargs=(recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_process),
    ) as pool:
        jobs = []
        for ind0, ind1 in zip(indices0, indices1):
            label0 = labels_set[ind0]
            label1 = labels_set[ind1]
            jobs.append(pool.submit(merge_function_wrapper, label0, label1))

        # jobs = jobs[5:15]
        # jobs = jobs[20:25]

        if progress_bar:
            iterator = tqdm(jobs, desc=f"merge_clusters with {method}", total=len(jobs))
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
    # print(indices0.size)

    return pair_mask, pair_shift




def merge_worker_init(recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_process):
    global _ctx
    _ctx = {}

    _ctx["recording"] = recording
    _ctx["original_labels"] = original_labels
    _ctx["method"] = method
    _ctx["method_kwargs"] = method_kwargs
    _ctx["method_class"] = merge_methods_dict[method]
    _ctx["max_threads_per_process"] = max_threads_per_process

    if isinstance(features_dict_or_folder, dict):
        _ctx["features"] = features_dict_or_folder
    else:
        _ctx["features"] = FeaturesLoader(features_dict_or_folder)
    _ctx["peaks"] = _ctx["features"]["peaks"]


def merge_function_wrapper(label0, label1):
    global _ctx
    with threadpool_limits(limits=_ctx["max_threads_per_process"]):
        is_merge, label0, label1, shift = _ctx["method_class"].merge(
            label0, label1, _ctx["original_labels"], _ctx["peaks"], _ctx["features"],
            **_ctx["method_kwargs"])
    return is_merge, label0, label1, shift


class WaveformsLda:
    name = 'waveforms_lda'

    @staticmethod
    def merge(label0, 
              label1,
              original_labels,
              peaks,
              features,
              waveforms_sparse_mask=None,
              feature_name="sparse_tsvd",
              projection="centroid",
              criteria="diptest",
              threshold_diptest=0.5,
              threshold_percentile=80.,

            #   neighbours_mask=None,
    ):

        sparse_wfs = features[feature_name]

        assert waveforms_sparse_mask is not None

        inds0, = np.nonzero(original_labels == label0)
        chans0 = np.unique(peaks["channel_index"][inds0])
        target_chans0 = np.flatnonzero(np.all(waveforms_sparse_mask[chans0, :], axis=0))

        inds1, = np.nonzero(original_labels == label1)
        chans1 = np.unique(peaks["channel_index"][inds1])
        target_chans1 = np.flatnonzero(np.all(waveforms_sparse_mask[chans1, :], axis=0))

        target_chans = np.intersect1d(target_chans0, target_chans1)
        # print()
        # print(label0, label1)
        # print(chans0, target_chans0)
        # print(chans1, target_chans1)
        # print(target_chans)


        # wfs0, out0 = aggregate_sparse_features(peaks, inds0, sparse_wfs, waveforms_sparse_mask, target_chans)
        # wfs1, out1 = aggregate_sparse_features(peaks, inds1, sparse_wfs, waveforms_sparse_mask, target_chans)

        # wfs0 = wfs0[~out0]
        # wfs1 = wfs1[~out1]
        inds = np.concatenate([inds0, inds1])
        labels = np.zeros(inds.size, dtype='int')
        labels[inds0.size:] = 1 
        # print(labels.shape)
        wfs, out = aggregate_sparse_features(peaks, inds, sparse_wfs, waveforms_sparse_mask, target_chans)
        # print(wfs.shape, out.shape)
        wfs = wfs[~out]
        labels = labels[~out]

        cut = np.searchsorted(labels, 1)
        # print('cut', cut, labels.size, inds0.size)
        wfs0 = wfs[:cut, :, :]
        wfs1 = wfs[cut:, :, :]


        if projection == "lda":
            flat_wfs = wfs.reshape(wfs.shape[0], -1)
            feat = LinearDiscriminantAnalysis(n_components=1).fit_transform(flat_wfs, labels)
            feat = feat[:, 0]
            feat0 = feat[:cut]
            feat1 = feat[cut:]

        elif projection == "centroid":
            template0 = np.mean(wfs0, axis=0)
            template1 = np.mean(wfs1, axis=0)
            vector_0_1 = template1 - template0
            vector_0_1 /= np.sum(vector_0_1**2)
            feat0 = np.sum((wfs0 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            feat1 = np.sum((wfs1 - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))
            feat  = np.sum((wfs - template0[np.newaxis, :, :]) * vector_0_1[np.newaxis, :, :], axis=(1, 2))

        else:
            raise ValueError(f"bad projection {projection}")
        
        if criteria == "diptest":
            dipscore, cutpoint = isocut5(feat)
            is_merge = dipscore < threshold_diptest
        elif criteria == "percentile":
            l0 = np.percentile(feat0, threshold_percentile)
            l1 = np.percentile(feat1, 100. - threshold_percentile)
            is_merge = l0 >= l1
        else:
            raise ValueError(f"bad criteria {criteria}")


        shift = 0

        # threshold_diptest=0.5,
        # dipscore, cutpoint = isocut5(lda_projs.squeeze())


        # dipscore, cutpoint = isocut5(lda_projs.squeeze())


        # print(wfs0.shape, wfs1.shape)

        # DEBUG = True
        DEBUG = False
        if DEBUG and is_merge:
            import matplotlib.pyplot as plt

            flatten_wfs0 = wfs0.swapaxes(1, 2).reshape(wfs0.shape[0], -1)
            flatten_wfs1 = wfs1.swapaxes(1, 2).reshape(wfs1.shape[0], -1)

            fig, axs = plt.subplots(ncols=2)
            ax = axs[0]
            ax.plot(flatten_wfs0.T, color='C0', alpha=.01)
            ax.plot(flatten_wfs1.T, color='C1', alpha=.01)
            m0 = np.mean(flatten_wfs0, axis=0)
            m1 = np.mean(flatten_wfs1, axis=0)
            ax.plot(m0, color='C0', alpha=1, lw=4)
            ax.plot(m1, color='C1', alpha=1, lw=4)


            bins = np.linspace(np.percentile(feat, 1), np.percentile(feat, 99), 100)

            count0, _ = np.histogram(feat0, bins=bins)
            count1, _ = np.histogram(feat1, bins=bins)


            ax = axs[1]
            ax.plot(bins[:-1], count0, color='C0')
            ax.plot(bins[:-1], count1, color='C1')

            ax.set_title(f'{dipscore}')






        # target channel subset is done intersect local channels + neighbours
        # local_chans_A = np.unique(peaks["channel_index"][peak_indices_A])
        # target_channels_A = np.flatnonzero(np.all(neighbours_mask[local_chans_A, :], axis=0))

        # local_chans_B = np.unique(peaks["channel_index"][peak_indices_B])
        # target_channels_B = np.flatnonzero(np.all(neighbours_mask[local_chans_B, :], axis=0))

        # print(label0, label1)


        return is_merge, label0, label1, shift


merge_methods_list = [
    WaveformsLda,
]
merge_methods_dict = {e.name: e for e in merge_methods_list}
