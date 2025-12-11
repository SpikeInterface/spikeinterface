from __future__ import annotations

from pathlib import Path
import numpy as np


from spikeinterface.core import Templates
from spikeinterface.core.recording_tools import get_channel_distances
from spikeinterface.sortingcomponents.waveforms.peak_svd import extract_peaks_svd
from spikeinterface.sortingcomponents.clustering.merging_tools import merge_peak_labels_from_templates
from spikeinterface.sortingcomponents.clustering.itersplit_tools import split_clusters
from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd


class IterativeHDBSCANClustering:
    """
    Iterative HDBSCAN is based on several local clustering achieved with a
    divide-and-conquer strategy. It uses the `hdbscan`clustering algorithms to
    perform the local clusterings with an iterative and greedy strategy.
    More precisely, it first extracts waveforms from the recording,
    then performs a Truncated SVD to reduce the dimensionality of the waveforms.
    For every peak, it extracts the SVD features and performs local clustering, grouping the peaks
    by channel indices. The clustering is done recursively, and the clusters are merged
    based on a similarity metric. The final output is a set of labels for each peak,
    indicating the cluster to which it belongs.
    """

    name = "iterative-hdbscan"
    need_noise_levels = False
    _default_params = {
        "peaks_svd": {"n_components": 5, "ms_before": 0.5, "ms_after": 1.5, "radius_um": 100.0},
        "seed": None,
        "split": {
            "split_radius_um": 75.0,
            "recursive": True,
            "recursive_depth": 3,
            "method_kwargs": {
                "clusterer": {
                    "method": "hdbscan",
                    "min_cluster_size": 20,
                    "allow_single_cluster": True,
                },
                "n_pca_features": 3,
            },
        },
        "merge_from_templates": dict(similarity_thresh=0.8, num_shifts=3, use_lags=True),
        "merge_from_features": None,
        "debug_folder": None,
        "verbose": True,
    }
    params_doc = """
        peaks_svd : params for peak SVD features extraction.
        See spikeinterface.sortingcomponents.waveforms.peak_svd.extract_peaks_svd
                        for more details.,
        seed : Random seed for reproducibility.,
        split : "params for the splitting step. See
                 spikeinterface.sortingcomponents.clustering.splitting_tools.split_clusters
                 for more details.,
        merge_from_templates : params for the merging step based on templates. See
                 spikeinterface.sortingcomponents.clustering.merging_tools.merge_peak_labels_from_templates
                 for more details.,
        merge_from_features : params for the merging step based on features. See
                    spikeinterface.sortingcomponents.clustering.merging_tools.merge_peak_labels_from_features
                    for more details.,
        debug_folder : If not None, a folder path where to save debug information.,
        verbose : If True, print information during the process.
    """

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        split_radius_um = params["split"].pop("split_radius_um", 75)
        peaks_svd = params["peaks_svd"]
        ms_before = peaks_svd["ms_before"]
        ms_after = peaks_svd["ms_after"]
        verbose = params["verbose"]
        min_cluster_size = params["split"]["method_kwargs"]["clusterer"]["min_cluster_size"]
        split = params["split"].copy()
        seed = params["seed"]
        debug_folder = params["debug_folder"]

        if debug_folder is not None:
            debug_folder = Path(debug_folder).absolute()
            debug_folder.mkdir(exist_ok=True)
            peaks_svd.update(features_folder=debug_folder / "features")

        if seed is not None:
            peaks_svd.update(seed=seed)
            split["method_kwargs"].update(seed=seed)

        peaks_svd, sparse_mask, svd_model = extract_peaks_svd(
            recording,
            peaks,
            job_kwargs=job_kwargs,
            **peaks_svd,
        )

        if debug_folder is not None:
            np.save(debug_folder / "sparse_mask.npy", sparse_mask)
            np.save(debug_folder / "peaks.npy", peaks)

        split["method_kwargs"].update(waveforms_sparse_mask=sparse_mask)
        neighbours_mask = get_channel_distances(recording) <= split_radius_um
        split["method_kwargs"].update(neighbours_mask=neighbours_mask)
        split["method_kwargs"].update(min_size_split=2 * min_cluster_size)

        if debug_folder is not None:
            split.update(debug_folder=debug_folder / "split")

        peak_labels = split_clusters(
            peaks["channel_index"],
            recording,
            {"peaks": peaks, "sparse_tsvd": peaks_svd},
            method="local_feature_clustering",
            job_kwargs=job_kwargs,
            **split,
        )

        templates, new_sparse_mask = get_templates_from_peaks_and_svd(
            recording,
            peaks,
            peak_labels,
            ms_before,
            ms_after,
            svd_model,
            peaks_svd,
            sparse_mask,
            operator="median",
        )

        labels = templates.unit_ids

        if verbose:
            print("Kept %d raw clusters" % len(labels))

        if params["merge_from_templates"] is not None:
            peak_labels, merge_template_array, new_sparse_mask, new_unit_ids = merge_peak_labels_from_templates(
                peaks,
                peak_labels,
                templates.unit_ids,
                templates.templates_array,
                new_sparse_mask,
                **params["merge_from_templates"],
            )

            templates = Templates(
                templates_array=merge_template_array,
                sampling_frequency=recording.sampling_frequency,
                nbefore=templates.nbefore,
                sparsity_mask=None,
                channel_ids=recording.channel_ids,
                unit_ids=new_unit_ids,
                probe=recording.get_probe(),
                is_in_uV=False,
            )

        labels = templates.unit_ids

        if debug_folder is not None:
            templates.to_zarr(folder_path=debug_folder / "dense_templates")

        if verbose:
            print("Kept %d non-duplicated clusters" % len(labels))

        more_outs = dict(
            svd_model=svd_model,
            peaks_svd=peaks_svd,
            peak_svd_sparse_mask=sparse_mask,
        )
        return labels, peak_labels, more_outs
