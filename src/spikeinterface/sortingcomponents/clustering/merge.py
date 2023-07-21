
from pathlib import Path
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from hdbscan import HDBSCAN

import numpy as np

from spikeinterface.core.job_tools import get_poolexecutor


from .tools import aggregate_sparse_features, FeaturesLoader

# This is not working at the moment

def merge_clusters(
    peak_labels,
    recording,
    features_dict_or_folder,
    method="waveforms_lda",
    method_kwargs={},
    recursive=False,

    n_jobs=1,
    mp_context="fork",
    max_threads_per_process=1,
    progress_bar=True,
):
    """
    Try some merges on clusters in parralel


    """
    # features_dict_or_folder = Path(features_dict_or_folder)

    original_labels = peak_labels
    peak_labels = peak_labels.copy()

    
    labels_set = np.setdiff1d(peak_labels, [-1])
    n = len(labels_set)
    pair_mask = np.triu(np.ones((n, n), dtype="bool"))

    # compute template
    # templates = np.zeros((n, ))
    # for i, label in labels_set:

    # wfs = aggregate_sparse_features(peaks, peak_indices, sparse_waveforms, sparse_mask, target_channels)
    # np.mean(wfs, axis=0)


    Executor = get_poolexecutor(n_jobs)

    with Executor(
        max_workers=n_jobs,
        initializer=merge_worker_init,
        mp_context=get_context(mp_context),
        initargs=(recording, features_dict_or_folder, original_labels, method, method_kwargs, max_threads_per_process),
    ) as pool:
        labels_set = np.setdiff1d(peak_labels, [-1])
        current_max_label = np.max(labels_set) + 1

        jobs = []
        for label in labels_set:
            peak_indices = np.flatnonzero(peak_labels == label)
            if peak_indices.size > 0:
                jobs.append(pool.submit(merge_function_wrapper, peak_indices))

        if progress_bar:
            iterator = tqdm(jobs, desc=f"split_clusters with {method}", total=len(labels_set))
        else:
            iterator = jobs

        for res in iterator:
            pass


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


def merge_function_wrapper(peak_indices):
    global _ctx
    with threadpool_limits(limits=_ctx["max_threads_per_process"]):
        is_split, local_labels = _ctx["method_class"].split(
            peak_indices, _ctx["peaks"], _ctx["features"], **_ctx["method_kwargs"]
        )
    return is_split, local_labels, peak_indices


class WaveformsLda:
    name = 'waveforms_lda'
    def merge(
        peak_indices_A,
        peak_indices_B,
        peaks,
        features,
        neighbours_mask=None,
        waveforms_sparse_mask=None,
        min_size_split=25,
        min_cluster_size=25,
        min_samples=25,
        n_pca_features=2,
    ):    

        sparse_wfs = features["sparse_wfs"]
        assert waveforms_sparse_mask is not None

        # target channel subset is done intersect local channels + neighbours
        local_chans_A = np.unique(peaks["channel_index"][peak_indices_A])
        target_channels_A = np.flatnonzero(np.all(neighbours_mask[local_chans_A, :], axis=0))

        local_chans_B = np.unique(peaks["channel_index"][peak_indices_B])
        target_channels_B = np.flatnonzero(np.all(neighbours_mask[local_chans_B, :], axis=0))




merge_methods_list = [
    WaveformsLda,
]
merge_methods_dict = {e.name: e for e in merge_methods_list}
