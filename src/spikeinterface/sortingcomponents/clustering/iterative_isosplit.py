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
from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd, remove_small_cluster
from spikeinterface.sortingcomponents.tools import clean_templates
from spikeinterface.sortingcomponents.waveforms.peak_svd import extract_peaks_svd
from spikeinterface.core.recording_tools import get_noise_levels


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
        "noise_levels": None,
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
                "n_pca_features": 6,
                # "n_pca_features": 10,
                "projection_mode": "tsvd",
                # "projection_mode": "pca",
            },
        },
        "clean_templates": {
            "max_jitter_ms": 0.2,
            "min_snr": 2.5,
            "sparsify_threshold": 1.0,
            "remove_empty": True,
        },
        "merge_from_templates": {
            "similarity_metric": "l1",
            "num_shifts": 3,
            "similarity_thresh": 0.8,
        },
        "merge_from_features": None,
        # "merge_from_features": {"merge_radius_um": 60.0},
        "clean_low_firing": {
            "min_firing_rate": 0.1,
            "subsampling_factor": None,
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
        verbose = params["verbose"]

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

        ## Pre clean using templates (jitter, sparsify_threshold)
        templates = dense_templates.to_sparse(template_sparse_mask)
        cleaning_kwargs = params["clean_templates"].copy()
        cleaning_kwargs["verbose"] = verbose
        # cleaning_kwargs["verbose"] = True
        # cleaning_kwargs["max_std_per_channel"] = max_std_per_channel
        if params["noise_levels"] is not None:
            noise_levels = params["noise_levels"]
        else:
            noise_levels = get_noise_levels(recording, return_in_uV=False, **job_kwargs)
        cleaning_kwargs["noise_levels"] = noise_levels
        cleaned_templates = clean_templates(templates, **cleaning_kwargs)
        mask_keep_ids = np.isin(templates.unit_ids, cleaned_templates.unit_ids)
        to_remove_ids = templates.unit_ids[~mask_keep_ids]
        to_remove_label_mask = np.isin(post_split_label, to_remove_ids)
        post_split_label[to_remove_label_mask] = -1
        template_sparse_mask = cleaned_templates.sparsity.mask.copy()
        dense_templates = cleaned_templates.to_dense()
        templates_array = dense_templates.templates_array
        unit_ids = dense_templates.unit_ids

        # ## Pre clean using templates (jitter)
        # cleaned_templates = clean_templates(
        #     dense_templates,
        #     # sparsify_threshold=0.25,
        #     sparsify_threshold=None,
        #     # noise_levels=None,
        #     # min_snr=None,
        #     max_jitter_ms=params["clean_templates"]["max_jitter_ms"],
        #     # remove_empty=True,
        #     remove_empty=False,
        #     # sd_ratio_threshold=5.0,
        #     # stds_at_peak=None,
        # )
        # mask_keep_ids = np.isin(dense_templates.unit_ids, cleaned_templates.unit_ids)
        # to_remove_ids = dense_templates.unit_ids[~mask_keep_ids]
        # to_remove_label_mask = np.isin(post_split_label, to_remove_ids)
        # post_split_label[to_remove_label_mask] = -1
        # dense_templates = cleaned_templates
        # template_sparse_mask = template_sparse_mask[mask_keep_ids, :]

        # unit_ids = dense_templates.unit_ids
        # templates_array = dense_templates.templates_array

        if params["merge_from_features"] is not None:

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

        # clean very small cluster before peeler
        if (
            params["clean_low_firing"]["subsampling_factor"] is not None
            and params["clean_low_firing"]["min_firing_rate"] is not None
        ):
            final_peak_labels, to_keep = remove_small_cluster(
                recording,
                peaks,
                post_merge_label2,
                min_firing_rate=params["clean_low_firing"]["min_firing_rate"],
                subsampling_factor=params["clean_low_firing"]["subsampling_factor"],
                verbose=verbose,
            )
            templates = templates.select_units(to_keep)
        else:
            final_peak_labels = post_merge_label2

        labels_set = templates.unit_ids

        more_outs = dict(
            templates=templates,
        )
        return labels_set, final_peak_labels, more_outs
