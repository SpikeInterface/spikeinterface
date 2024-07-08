from pathlib import Path

import numpy as np
import json
import pickle
import random
import string
import shutil

from spikeinterface.core import (
    get_channel_distances,
    Templates,
    compute_sparsity,
    get_global_tmp_folder,
)

from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    ExtractDenseWaveforms,
    ExtractSparseWaveforms,
    PeakRetriever,
)

from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel, cache_preprocessing
from spikeinterface.sortingcomponents.peak_detection import detect_peaks, DetectPeakLocallyExclusive
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass, LocalizeGridConvolution
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection

from spikeinterface.sortingcomponents.clustering.split import split_clusters
from spikeinterface.sortingcomponents.clustering.merge import merge_clusters
from spikeinterface.sortingcomponents.clustering.tools import compute_template_from_sparse


class TdcClustering:
    """
    Here the implementation of clustering used by tridesclous2
    """

    _default_params = {
        "folder": None,
        "waveforms": {
            "ms_before": 0.5,
            "ms_after": 1.5,
            "radius_um": 120.0,
        },
        "svd": {"n_components": 6},
        "clustering": {
            "split_radius_um": 40.0,
            "merge_radius_um": 40.0,
            "threshold_diff": 1.5,
        },
        "job_kwargs": {},
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        import hdbscan

        job_kwargs = params["job_kwargs"]

        if params["folder"] is None:
            randname = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            clustering_folder = get_global_tmp_folder() / f"tdcclustering_{randname}"
            clustering_folder.mkdir(parents=True, exist_ok=True)
            need_folder_rm = True
        else:
            clustering_folder = Path(params["folder"])
            need_folder_rm = False

        sampling_frequency = recording.sampling_frequency

        ms_before = params["waveforms"]["ms_before"]
        ms_after = params["waveforms"]["ms_after"]

        nbefore = int(ms_before * sampling_frequency / 1000.0)
        nafter = int(ms_after * sampling_frequency / 1000.0)

        # SVD for time compression
        few_peaks = select_peaks(peaks, recording=recording, method="uniform", n_peaks=5000, margin=(nbefore, nafter))
        few_wfs = extract_waveform_at_max_channel(
            recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
        )

        wfs = few_wfs[:, :, 0]
        from sklearn.decomposition import TruncatedSVD

        tsvd = TruncatedSVD(params["svd"]["n_components"])
        tsvd.fit(wfs)

        model_folder = clustering_folder / "tsvd_model"

        model_folder.mkdir(exist_ok=True)
        with open(model_folder / "pca_model.pkl", "wb") as f:
            pickle.dump(tsvd, f)

        model_params = {
            "ms_before": ms_before,
            "ms_after": ms_after,
            "sampling_frequency": float(sampling_frequency),
        }
        with open(model_folder / "params.json", "w") as f:
            json.dump(model_params, f)

        # features

        features_folder = clustering_folder / "features"
        node0 = PeakRetriever(recording, peaks)

        radius_um = params["waveforms"]["radius_um"]
        node1 = ExtractSparseWaveforms(
            recording,
            parents=[node0],
            return_output=True,
            ms_before=ms_before,
            ms_after=ms_after,
            radius_um=radius_um,
        )

        model_folder_path = clustering_folder / "tsvd_model"

        node2 = TemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, model_folder_path=model_folder_path
        )

        pipeline_nodes = [node0, node1, node2]

        run_node_pipeline(
            recording,
            pipeline_nodes,
            job_kwargs,
            gather_mode="npy",
            gather_kwargs=dict(exist_ok=True),
            folder=features_folder,
            names=["sparse_wfs", "sparse_tsvd"],
        )
        # TODO make this generic in GatherNPY ???
        sparse_mask = node1.neighbours_mask
        np.save(features_folder / "sparse_mask.npy", sparse_mask)
        np.save(features_folder / "peaks.npy", peaks)

        # to be able to delete feature folder
        del pipeline_nodes, node0, node1, node2

        # Clustering: channel index > split > merge
        split_radius_um = params["clustering"]["split_radius_um"]
        neighbours_mask = get_channel_distances(recording) < split_radius_um

        original_labels = peaks["channel_index"]

        min_cluster_size = 50
        # min_cluster_size = 10

        post_split_label, split_count = split_clusters(
            original_labels,
            recording,
            features_folder,
            method="local_feature_clustering",
            method_kwargs=dict(
                clusterer="hdbscan",
                clusterer_kwargs={
                    "min_cluster_size": min_cluster_size,
                    "allow_single_cluster": True,
                    "cluster_selection_method": "eom",
                },
                # clusterer="isocut5",
                # clusterer_kwargs={"min_cluster_size": min_cluster_size},
                feature_name="sparse_tsvd",
                # feature_name="sparse_wfs",
                neighbours_mask=neighbours_mask,
                waveforms_sparse_mask=sparse_mask,
                min_size_split=min_cluster_size,
                n_pca_features=3,
                scale_n_pca_by_depth=True,
            ),
            recursive=True,
            recursive_depth=3,
            returns_split_count=True,
            **job_kwargs,
        )

        merge_radius_um = params["clustering"]["merge_radius_um"]
        threshold_diff = params["clustering"]["threshold_diff"]

        post_merge_label, peak_shifts = merge_clusters(
            peaks,
            post_split_label,
            recording,
            features_folder,
            radius_um=merge_radius_um,
            # method="project_distribution",
            # method_kwargs=dict(
            #     waveforms_sparse_mask=sparse_mask,
            #     feature_name="sparse_wfs",
            #     projection="centroid",
            #     criteria="distrib_overlap",
            #     threshold_overlap=0.3,
            #     min_cluster_size=min_cluster_size + 1,
            #     num_shift=5,
            # ),
            method="normalized_template_diff",
            method_kwargs=dict(
                waveforms_sparse_mask=sparse_mask,
                threshold_diff=threshold_diff,
                min_cluster_size=min_cluster_size + 1,
                num_shift=5,
            ),
            **job_kwargs,
        )

        # sparse_wfs = np.load(features_folder / "sparse_wfs.npy", mmap_mode="r")

        # new_peaks = peaks.copy()
        # new_peaks["sample_index"] -= peak_shifts

        # clean very small cluster before peeler
        post_clean_label = post_merge_label.copy()

        minimum_cluster_size = 25
        labels_set, count = np.unique(post_clean_label, return_counts=True)
        to_remove = labels_set[count < minimum_cluster_size]
        mask = np.isin(post_clean_label, to_remove)
        post_clean_label[mask] = -1

        labels_set = np.unique(post_clean_label)
        labels_set = labels_set[labels_set >= 0]

        if need_folder_rm:
            shutil.rmtree(clustering_folder)

        extra_out = {"peak_shifts": peak_shifts}
        return labels_set, post_clean_label, extra_out
