from pathlib import Path
import importlib
import numpy as np

from spikeinterface.core import get_channel_distances, Templates, ChannelSparsity
from spikeinterface.sortingcomponents.clustering.itersplit_tools import split_clusters

# from spikeinterface.sortingcomponents.clustering.merge import merge_clusters
from spikeinterface.sortingcomponents.clustering.merging_tools import (
    merge_peak_labels_from_templates,
    merge_peak_labels_from_features,
)
from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd
from spikeinterface.sortingcomponents.waveforms.peak_svd import extract_peaks_svd


class IterativeISOSPLITClustering:
    """
    Iterative ISOSPLIT is based on several local clustering achieved with a
    divide-and-conquer strategy. It uses the `isosplit`clustering algorithms to
    perform the local clusterings with an iterative and greedy strategy.
    More precisely, it first extracts waveforms from the recording,
    then performs a Truncated SVD to reduce the dimensionality of the waveforms.
    For every peak, it extracts the SVD features and performs local clustering, grouping the peaks
    by channel indices. The clustering is done recursively, and the clusters are merged
    based on a similarity metric. The final output is a set of labels for each peak,
    indicating the cluster to which it belongs.
    """

    name = "iterative-isosplit"
    need_noise_levels = False
    _default_params = {
        "motion": None,
        "seed": None,
        "peaks_svd": {"n_components": 5, "ms_before": 0.5, "ms_after": 1.5, "radius_um": 120.0, "motion": None},
        "pre_label": {
            "mode": "channel",
            # "mode": "vertical_bin",
        },
        "split": {
            # "split_radius_um": 40.0,
            "split_radius_um": 60.0,
            "recursive": True,
            "recursive_depth": 3,
            "method_kwargs": {
                "clusterer": {
                    "method": "isosplit",
                    # "method": "isosplit6",
                    # "n_init": 50,
                    "min_cluster_size": 10,
                    "max_iterations_per_pass": 500,
                    "isocut_threshold": 2.0,
                    # "isocut_threshold": 2.2,
                },
                "min_size_split": 25,
                # "n_pca_features": 3,
                "n_pca_features": 10,
                "projection_mode": "tsvd",
                # "projection_mode": "pca",
            },
        },
        "merge_from_templates": {
            "similarity_metric": "l1",
            "num_shifts": 3,
            "similarity_thresh": 0.8,
        },
        "merge_from_features": None,
        # "merge_from_features": {},
        "clean": {
            "minimum_cluster_size": 10,
        },
        "debug_folder": None,
        "verbose": True,
    }

    params_doc = """
        peaks_svd : params for peak SVD features extraction.
        See spikeinterface.sortingcomponents.waveforms.peak_svd.extract_peaks_svd
                        for more details.,
        seed : Random seed for reproducibility.,
        split : params for the splitting step. See
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

        ms_before = params["peaks_svd"]["ms_before"]
        ms_after = params["peaks_svd"]["ms_after"]
        # radius_um = params["waveforms"]["radius_um"]

        debug_folder = params["debug_folder"]

        params_peak_svd = params["peaks_svd"].copy()

        motion = params_peak_svd["motion"]
        motion_aware = motion is not None

        # extract svd
        outs = extract_peaks_svd(
            recording,
            peaks,
            job_kwargs=job_kwargs,
            motion_aware=motion_aware,
            **params_peak_svd,
        )

        if motion is not None:
            # also return peaks with new channel index
            peaks_svd, sparse_mask, svd_model, moved_peaks = outs
            peaks = moved_peaks
        else:
            peaks_svd, sparse_mask, svd_model = outs

        # Clustering: channel index > split > merge
        split_params = params["split"].copy()

        split_radius_um = split_params.pop("split_radius_um")
        neighbours_mask = get_channel_distances(recording) <= split_radius_um
        split_params["method_kwargs"]["neighbours_mask"] = neighbours_mask
        split_params["method_kwargs"]["waveforms_sparse_mask"] = sparse_mask
        split_params["method_kwargs"]["feature_name"] = "peaks_svd"

        if params["pre_label"]["mode"] == "channel":
            original_labels = peaks["channel_index"]
        elif params["pre_label"]["mode"] == "vertical_bin":
            # 2 params
            direction = "y"
            bin_um = 40.0

            channel_locations = recording.get_channel_locations()
            dim = "xyz".index(direction)
            channel_depth = channel_locations[:, dim]

            # bins
            min_ = np.min(channel_depth)
            max_ = np.max(channel_depth)
            num_windows = int((max_ - min_) // bin_um)
            num_windows = max(num_windows, 1)
            border = ((max_ - min_) % bin_um) / 2
            vertical_bins = np.zeros(num_windows + 3)
            vertical_bins[1:-1] = np.arange(num_windows + 1) * bin_um + min_ + border
            vertical_bins[0] = -np.inf
            vertical_bins[-1] = np.inf
            # peak depth
            peak_depths = channel_depth[peaks["channel_index"]]
            # label by bin
            original_labels = np.digitize(peak_depths, vertical_bins)

        # clusterer = params["split"]["clusterer"]
        # clusterer_kwargs = params["split"]["clusterer_kwargs"]

        features = dict(
            peaks=peaks,
            peaks_svd=peaks_svd,
        )

        split_params["returns_split_count"] = True

        if params["seed"] is not None:
            split_params["method_kwargs"]["clusterer"]["seed"] = params["seed"]

        post_split_label, split_count = split_clusters(
            original_labels,
            recording,
            # features_folder,
            features,
            method="local_feature_clustering",
            debug_folder=debug_folder,
            job_kwargs=job_kwargs,
            # job_kwargs=dict(n_jobs=1),
            **split_params,
            # method_kwargs=dict(
            #     clusterer=clusterer,
            #     clusterer_kwargs=clusterer_kwargs,
            #     feature_name="peaks_svd",
            #     neighbours_mask=neighbours_mask,
            #     waveforms_sparse_mask=sparse_mask,
            #     min_size_split=params["split"]["min_size_split"],
            #     n_pca_features=3,
            # ),
            # recursive=True,
            # recursive_depth=params["split"]["recursive_depth"],
            # returns_split_count=True,
            # debug_folder=clustering_folder / "figure_debug_split",
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

        if params["merge_from_features"]:

            merge_from_features_kwargs = params["merge_from_features"].copy()
            merge_radius_um = merge_from_features_kwargs.pop("merge_radius_um")

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
                    feature_name="peaks_svd", waveforms_sparse_mask=sparse_mask, **merge_from_features_kwargs
                ),
                job_kwargs=job_kwargs,
            )
        else:
            post_merge_label1 = post_split_label.copy()

        if params["merge_from_templates"] is not None:
            post_merge_label2, templates_array, template_sparse_mask, unit_ids = merge_peak_labels_from_templates(
                peaks,
                post_merge_label1,
                unit_ids,
                templates_array,
                template_sparse_mask,
                **params["merge_from_templates"],
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

    # _default_params = {
    #     "clean": {
    #         "minimum_cluster_size": 10,
    #     },
    # }

    # @classmethod
    # def main_function(cls, recording, peaks, params, job_kwargs=dict()):

    #     split_radius_um = params["split"].pop("split_radius_um", 40)
    #     peaks_svd = params["peaks_svd"]
    #     motion = peaks_svd["motion"]
    #     ms_before = peaks_svd.get("ms_before", 0.5)
    #     ms_after = peaks_svd.get("ms_after", 1.5)
    #     verbose = params.get("verbose", True)
    #     split = params["split"]
    #     seed = params["seed"]
    #     job_kwargs = params.get("job_kwargs", dict())
    #     debug_folder = params.get("debug_folder", None)

    #     if debug_folder is not None:
    #         debug_folder = Path(debug_folder).absolute()
    #         debug_folder.mkdir(exist_ok=True)
    #         peaks_svd.update(folder=debug_folder / "features")

    #     motion_aware = motion is not None
    #     peaks_svd.update(motion_aware=motion_aware)

    #     if seed is not None:
    #         peaks_svd.update(seed=seed)
    #         split["method_kwargs"].update(seed=seed)

    #     outs = extract_peaks_svd(
    #         recording,
    #         peaks,
    #         **peaks_svd,
    #         **job_kwargs,
    #     )

    #     if motion_aware:
    #         # also return peaks with new channel index
    #         peaks_svd, sparse_mask, svd_model, moved_peaks = outs
    #         peaks = moved_peaks
    #     else:
    #         peaks_svd, sparse_mask, svd_model = outs

    #     if debug_folder is not None:
    #         np.save(debug_folder / "sparse_mask.npy", sparse_mask)
    #         np.save(debug_folder / "peaks.npy", peaks)

    #     split["method_kwargs"].update(waveforms_sparse_mask = sparse_mask)
    #     neighbours_mask = get_channel_distances(recording) <= split_radius_um
    #     split["method_kwargs"].update(neighbours_mask = neighbours_mask)

    #     if debug_folder is not None:
    #         split.update(debug_folder = debug_folder / "split")

    #     peak_labels = split_clusters(
    #         peaks["channel_index"],
    #         recording,
    #         {"peaks": peaks, "sparse_tsvd": peaks_svd},
    #         method="local_feature_clustering",
    #         **split,
    #         **job_kwargs,
    #     )

    #     templates, new_sparse_mask = get_templates_from_peaks_and_svd(
    #         recording,
    #         peaks,
    #         peak_labels,
    #         ms_before,
    #         ms_after,
    #         svd_model,
    #         peaks_svd,
    #         sparse_mask,
    #         operator="average",
    #     )

    #     labels = templates.unit_ids

    #     if verbose:
    #         print("Kept %d raw clusters" % len(labels))

    #     if params["merge_from_features"] is not None:

    #         merge_features_kwargs = params["merge_from_features"].copy()
    #         merge_radius_um = merge_features_kwargs.pop("merge_radius_um")

    #         peak_labels, merge_template_array, new_sparse_mask, new_unit_ids = merge_peak_labels_from_features(
    #             peaks,
    #             peak_labels,
    #             templates.unit_ids,
    #             templates.templates_array,
    #             sparse_mask,
    #             recording,
    #             {"peaks": peaks, "sparse_tsvd": peaks_svd},
    #             radius_um=merge_radius_um,
    #             method="project_distribution",
    #             method_kwargs=dict(
    #                 feature_name="sparse_tsvd",
    #                 waveforms_sparse_mask=sparse_mask,
    #                 **merge_features_kwargs
    #             ),
    #             **job_kwargs,
    #         )

    #         templates = Templates(
    #             templates_array=merge_template_array,
    #             sampling_frequency=recording.sampling_frequency,
    #             nbefore=templates.nbefore,
    #             sparsity_mask=None,
    #             channel_ids=recording.channel_ids,
    #             unit_ids=new_unit_ids,
    #             probe=recording.get_probe(),
    #             is_in_uV=False,
    #         )

    #     if params["merge_from_templates"] is not None:
    #         peak_labels, merge_template_array, new_sparse_mask, new_unit_ids = merge_peak_labels_from_templates(
    #             peaks,
    #             peak_labels,
    #             templates.unit_ids,
    #             templates.templates_array,
    #             new_sparse_mask,
    #             **params["merge_from_templates"],
    #         )

    #         templates = Templates(
    #             templates_array=merge_template_array,
    #             sampling_frequency=recording.sampling_frequency,
    #             nbefore=templates.nbefore,
    #             sparsity_mask=None,
    #             channel_ids=recording.channel_ids,
    #             unit_ids=new_unit_ids,
    #             probe=recording.get_probe(),
    #             is_in_uV=False,
    #         )

    #     labels = templates.unit_ids

    #     if debug_folder is not None:
    #         templates.to_zarr(folder_path=debug_folder / "dense_templates")

    #     if verbose:
    #         print("Kept %d non-duplicated clusters" % len(labels))

    #     # sparsity = ChannelSparsity(template_sparse_mask, unit_ids, recording.channel_ids)
    #     # templates = dense_templates.to_sparse(sparsity)

    #     # # sparse_wfs = np.load(features_folder / "sparse_wfs.npy", mmap_mode="r")

    #     # # new_peaks = peaks.copy()
    #     # # new_peaks["sample_index"] -= peak_shifts

    #     # # clean very small cluster before peeler
    #     # post_clean_label = post_merge_label2.copy()
    #     # minimum_cluster_size = params["clean"]["minimum_cluster_size"]
    #     # labels_set, count = np.unique(post_clean_label, return_counts=True)
    #     # to_remove = labels_set[count < minimum_cluster_size]
    #     # mask = np.isin(post_clean_label, to_remove)
    #     # post_clean_label[mask] = -1
    #     # final_peak_labels = post_clean_label
    #     # labels_set = np.unique(final_peak_labels)
    #     # labels_set = labels_set[labels_set >= 0]
    #     # templates = templates.select_units(labels_set)
    #     # labels_set = templates.unit_ids

    #     more_outs = dict(
    #         templates=templates,
    #     )
    #     return labels, peak_labels, more_outs
