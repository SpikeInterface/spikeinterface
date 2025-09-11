from pathlib import Path

import numpy as np
import json
import pickle
import random
import string
import shutil

from spikeinterface.core import get_channel_distances, get_global_tmp_folder, Templates, ChannelSparsity

from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    ExtractSparseWaveforms,
    PeakRetriever,
)

from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection

from spikeinterface.sortingcomponents.clustering.split import split_clusters

# from spikeinterface.sortingcomponents.clustering.merge import merge_clusters
from spikeinterface.sortingcomponents.clustering.merge import (
    merge_peak_labels_from_templates,
    merge_peak_labels_from_features,
)
from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd
from spikeinterface.sortingcomponents.clustering.peak_svd import extract_peaks_svd


class TdcClustering:
    """
    Here the implementation of clustering used by tridesclous2
    """

    _default_params = {
        "motion": None,
        "seed": None,
        # "folder": None,
        "waveforms": {
            "ms_before": 0.5,
            "ms_after": 1.5,
            "radius_um": 120.0,
        },
        "extract_peaks_svd_kwargs": dict(n_components=5),
        "split": {
            "recursive_depth": 3,
            "split_radius_um": 40.0,
            # "clusterer": "hdbscan",
            # "clusterer_kwargs": {
            #     "min_cluster_size": 10,
            #     "min_samples": 1,
            #     "allow_single_cluster": True,
            #     "cluster_selection_method": "eom",
            # },
            # "clusterer": "isosplit6",
            # "clusterer_kwargs": {},
            "clusterer": "isosplit",
            "clusterer_kwargs": {
                "n_init": 50,
                "min_cluster_size": 10,
                "max_iterations_per_pass": 500,
                "isocut_threshold": 2.0,
            },
            "do_merge": True,
            "merge_kwargs": {
                "similarity_metric": "l1",
                "num_shifts": 3,
                "similarity_thresh": 0.8,
            },
            "min_size_split": 10,
        },
        "do_merge_with_features": False,
        "merge_features_kwargs": {"merge_radius_um": 50.0, "criteria": "isocut", "isocut_threshold": 2.0},
        "do_merge_with_templates": True,
        "merge_template_kwargs": {
            "similarity_metric": "l1",
            "num_shifts": 3,
            "similarity_thresh": 0.8,
        },
        "clean": {
            "minimum_cluster_size": 10,
        },
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        ms_before = params["waveforms"]["ms_before"]
        ms_after = params["waveforms"]["ms_after"]
        radius_um = params["waveforms"]["radius_um"]

        motion = params["motion"]
        motion_aware = motion is not None

        # extract svd
        outs = extract_peaks_svd(
            recording,
            peaks,
            ms_before=ms_before,
            ms_after=ms_after,
            radius_um=radius_um,
            motion_aware=motion_aware,
            motion=motion,
            **params["extract_peaks_svd_kwargs"],
            **job_kwargs,
        )

        if motion is not None:
            # also return peaks with new channel index
            peaks_svd, sparse_mask, svd_model, moved_peaks = outs
            peaks = moved_peaks
        else:
            peaks_svd, sparse_mask, svd_model = outs

        # Clustering: channel index > split > merge
        split_radius_um = params["split"]["split_radius_um"]
        neighbours_mask = get_channel_distances(recording) <= split_radius_um

        original_labels = peaks["channel_index"]

        clusterer = params["split"]["clusterer"]
        clusterer_kwargs = params["split"]["clusterer_kwargs"]

        features = dict(
            peaks=peaks,
            peaks_svd=peaks_svd,
        )

        post_split_label, split_count = split_clusters(
            original_labels,
            recording,
            # features_folder,
            features,
            method="local_feature_clustering",
            method_kwargs=dict(
                clusterer=clusterer,
                clusterer_kwargs=clusterer_kwargs,
                feature_name="peaks_svd",
                neighbours_mask=neighbours_mask,
                waveforms_sparse_mask=sparse_mask,
                min_size_split=params["split"]["min_size_split"],
                n_pca_features=3,
            ),
            recursive=True,
            recursive_depth=params["split"]["recursive_depth"],
            returns_split_count=True,
            # debug_folder=clustering_folder / "figure_debug_split",
            debug_folder=None,
            **job_kwargs,
        )

        dense_templates, template_sparse_mask = get_templates_from_peaks_and_svd(
            recording,
            peaks,
            post_split_label,
            ms_before,
            ms_after,
            svd_model,
            peaks_svd,
            sparse_mask,
            operator="average",
        )

        unit_ids = dense_templates.unit_ids
        templates_array = dense_templates.templates_array

        if params["do_merge_with_features"]:

            merge_features_kwargs = params["merge_features_kwargs"].copy()
            merge_radius_um = merge_features_kwargs.pop("merge_radius_um")

            post_merge_label1, templates_array, template_sparse_mask, unit_ids = merge_peak_labels_from_features(
                peaks,
                post_split_label,
                unit_ids,
                templates_array,
                template_sparse_mask,
                recording,
                features,
                radius_um=merge_radius_um,
                method="project_distribution",
                method_kwargs=dict(
                    feature_name="peaks_svd", waveforms_sparse_mask=sparse_mask, **merge_features_kwargs
                ),
                **job_kwargs,
            )
        else:
            post_merge_label1 = post_split_label.copy()

        if params["do_merge_with_templates"]:
            post_merge_label2, templates_array, template_sparse_mask, unit_ids = merge_peak_labels_from_templates(
                peaks,
                post_merge_label1,
                unit_ids,
                templates_array,
                template_sparse_mask,
                **params["merge_template_kwargs"],
            )
        else:
            post_merge_label2 = post_merge_label1.copy()

        dense_templates = Templates(
            templates_array=templates_array,
            sampling_frequency=dense_templates.sampling_frequency,
            nbefore=dense_templates.nbefore,
            sparsity_mask=None,
            channel_ids=recording.channel_ids,
            unit_ids=unit_ids,
            probe=recording.get_probe(),
            is_in_uV=False,
        )

        sparsity = ChannelSparsity(template_sparse_mask, unit_ids, recording.channel_ids)
        templates = dense_templates.to_sparse(sparsity)

        # sparse_wfs = np.load(features_folder / "sparse_wfs.npy", mmap_mode="r")

        # new_peaks = peaks.copy()
        # new_peaks["sample_index"] -= peak_shifts

        # clean very small cluster before peeler
        post_clean_label = post_merge_label2.copy()
        minimum_cluster_size = params["clean"]["minimum_cluster_size"]
        labels_set, count = np.unique(post_clean_label, return_counts=True)
        to_remove = labels_set[count < minimum_cluster_size]
        mask = np.isin(post_clean_label, to_remove)
        post_clean_label[mask] = -1
        final_peak_labels = post_clean_label
        labels_set = np.unique(final_peak_labels)
        labels_set = labels_set[labels_set >= 0]
        templates = templates.select_units(labels_set)
        labels_set = templates.unit_ids

        more_outs = dict(
            templates=templates,
        )
        return labels_set, final_peak_labels, more_outs
