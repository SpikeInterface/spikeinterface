from __future__ import annotations

import shutil

from .si_based import ComponentsBasedSorter

from spikeinterface.core import (
    get_noise_levels,
    NumpySorting,
    get_channel_distances,
    estimate_templates_with_accumulator,
    Templates,
    compute_sparsity,
)

from spikeinterface.core.job_tools import fix_job_kwargs

from spikeinterface.preprocessing import bandpass_filter, common_reference, zscore, whiten
from spikeinterface.core.basesorting import minimum_spike_dtype

from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel, cache_preprocessing

# from spikeinterface.qualitymetrics import compute_snrs

import numpy as np

import pickle
import json


class Tridesclous2Sorter(ComponentsBasedSorter):
    sorter_name = "tridesclous2"

    _default_params = {
        "apply_preprocessing": True,
        "cache_preprocessing": {"mode": "memory", "memory_limit": 0.5, "delete_cache": True},
        "waveforms": {
            "ms_before": 0.5,
            "ms_after": 1.5,
            "radius_um": 120.0,
        },
        "filtering": {"freq_min": 300.0, "freq_max": 12000.0},
        "detection": {"peak_sign": "neg", "detect_threshold": 5, "exclude_sweep_ms": 1.5, "radius_um": 150.0},
        "selection": {"n_peaks_per_channel": 5000, "min_n_peaks": 20000},
        "svd": {"n_components": 6},
        "clustering": {
            "split_radius_um": 40.0,
            "merge_radius_um": 40.0,
            "threshold_diff": 1.5,
        },
        "templates": {
            "ms_before": 2.0,
            "ms_after": 3.0,
            "max_spikes_per_unit": 400,
            # "peak_shift_ms": 0.2,
        },
        # "matching": {"method": "tridesclous", "method_kwargs": {"peak_shift_ms": 0.2, "radius_um": 100.0}},
        "matching": {"method": "circus-omp-svd", "method_kwargs": {}},
        "job_kwargs": {"n_jobs": -1},
        "save_array": True,
    }

    _params_description = {
        "apply_preprocessing": "Apply internal preprocessing or not",
        "cache_preprocessing": "A dict contaning how to cache the preprocessed recording. mode='memory' | 'folder | 'zarr' ",
        "waveforms": "A dictonary containing waveforms params: ms_before, ms_after, radius_um",
        "filtering": "A dictonary containing filtering params: freq_min, freq_max",
        "detection": "A dictonary containing detection params: peak_sign, detect_threshold, exclude_sweep_ms, radius_um",
        "selection": "A dictonary containing selection params: n_peaks_per_channel, min_n_peaks",
        "svd": "A dictonary containing svd params: n_components",
        "clustering": "A dictonary containing clustering params: split_radius_um, merge_radius_um",
        "templates": "A dictonary containing waveforms params for peeler: ms_before, ms_after",
        "matching": "A dictonary containing matching params for matching: peak_shift_ms, radius_um",
        "job_kwargs": "A dictionary containing job kwargs",
        "save_array": "Save or not intermediate arrays",
    }

    handle_multi_segment = True

    @classmethod
    def get_sorter_version(cls):
        return "2.0"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = verbose

        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.core.node_pipeline import (
            run_node_pipeline,
            ExtractDenseWaveforms,
            ExtractSparseWaveforms,
            PeakRetriever,
        )
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks, DetectPeakLocallyExclusive
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.peak_localization import LocalizeCenterOfMass, LocalizeGridConvolution
        from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection

        from spikeinterface.sortingcomponents.clustering.split import split_clusters
        from spikeinterface.sortingcomponents.clustering.merge import merge_clusters
        from spikeinterface.sortingcomponents.clustering.tools import compute_template_from_sparse

        from sklearn.decomposition import TruncatedSVD

        import hdbscan

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        # preprocessing
        if params["apply_preprocessing"]:
            recording = bandpass_filter(recording_raw, **params["filtering"])
            # TODO what is the best about zscore>common_reference or the reverse
            recording = common_reference(recording)
            recording = zscore(recording, dtype="float32")
            # recording = whiten(recording, dtype="float32")

            # used only if "folder" or "zarr"
            cache_folder = sorter_output_folder / "cache_preprocessing"
            recording = cache_preprocessing(
                recording, folder=cache_folder, **job_kwargs, **params["cache_preprocessing"]
            )

            noise_levels = np.ones(num_chans, dtype="float32")
        else:
            recording = recording_raw
            noise_levels = get_noise_levels(recording, return_scaled=False)

        # detection
        detection_params = params["detection"].copy()
        detection_params["noise_levels"] = noise_levels
        all_peaks = detect_peaks(recording, method="locally_exclusive", **detection_params, **job_kwargs)

        if verbose:
            print("We found %d peaks in total" % len(all_peaks))

        # selection
        selection_params = params["selection"].copy()
        n_peaks = params["selection"]["n_peaks_per_channel"] * num_chans
        n_peaks = max(selection_params["min_n_peaks"], n_peaks)
        peaks = select_peaks(all_peaks, method="uniform", n_peaks=n_peaks)

        if verbose:
            print("We kept %d peaks for clustering" % len(peaks))

        ms_before = params["waveforms"]["ms_before"]
        ms_after = params["waveforms"]["ms_after"]

        # SVD for time compression
        few_peaks = select_peaks(peaks, method="uniform", n_peaks=5000)
        few_wfs = extract_waveform_at_max_channel(
            recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
        )

        wfs = few_wfs[:, :, 0]
        tsvd = TruncatedSVD(params["svd"]["n_components"])
        tsvd.fit(wfs)

        model_folder = sorter_output_folder / "tsvd_model"

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

        features_folder = sorter_output_folder / "features"
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

        model_folder_path = sorter_output_folder / "tsvd_model"

        node2 = TemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, model_folder_path=model_folder_path
        )

        pipeline_nodes = [node0, node1, node2]

        output = run_node_pipeline(
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

        # Clustering: channel index > split > merge
        split_radius_um = params["clustering"]["split_radius_um"]
        neighbours_mask = get_channel_distances(recording) < split_radius_um

        original_labels = peaks["channel_index"]

        min_cluster_size = 50

        post_split_label, split_count = split_clusters(
            original_labels,
            recording,
            features_folder,
            method="local_feature_clustering",
            method_kwargs=dict(
                # clusterer="hdbscan",
                clusterer="isocut5",
                feature_name="sparse_tsvd",
                # feature_name="sparse_wfs",
                neighbours_mask=neighbours_mask,
                waveforms_sparse_mask=sparse_mask,
                min_size_split=min_cluster_size,
                clusterer_kwargs={"min_cluster_size": min_cluster_size},
                n_pca_features=3,
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

        new_peaks = peaks.copy()
        new_peaks["sample_index"] -= peak_shifts

        # clean very small cluster before peeler
        post_clean_label = post_merge_label.copy()

        minimum_cluster_size = 25
        labels_set, count = np.unique(post_clean_label, return_counts=True)
        to_remove = labels_set[count < minimum_cluster_size]
        mask = np.isin(post_clean_label, to_remove)
        post_clean_label[mask] = -1

        # final label sets
        labels_set = np.unique(post_clean_label)
        labels_set = labels_set[labels_set >= 0]

        mask = post_clean_label >= 0
        sorting_pre_peeler = NumpySorting.from_times_labels(
            new_peaks["sample_index"][mask],
            post_merge_label[mask],
            sampling_frequency,
            unit_ids=labels_set,
        )
        # sorting_pre_peeler = sorting_pre_peeler.save(folder=sorter_output_folder / "sorting_pre_peeler")

        recording_w = whiten(recording, mode="local", radius_um=100.0)

        nbefore = int(params["templates"]["ms_before"] * sampling_frequency / 1000.0)
        nafter = int(params["templates"]["ms_after"] * sampling_frequency / 1000.0)
        templates_array = estimate_templates_with_accumulator(
            recording_w,
            sorting_pre_peeler.to_spike_vector(),
            sorting_pre_peeler.unit_ids,
            nbefore,
            nafter,
            return_scaled=False,
            **job_kwargs,
        )
        templates_dense = Templates(
            templates_array=templates_array,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            probe=recording_w.get_probe(),
        )
        # TODO : try other methods for sparsity
        # sparsity = compute_sparsity(templates_dense, method="radius", radius_um=120.)
        sparsity = compute_sparsity(templates_dense, noise_levels=noise_levels, threshold=1.0)
        templates = templates_dense.to_sparse(sparsity)

        # snrs = compute_snrs(we, peak_sign=params["detection"]["peak_sign"], peak_mode="extremum")
        # print(snrs)

        # matching_params = params["matching"].copy()
        # matching_params["noise_levels"] = noise_levels
        # matching_params["peak_sign"] = params["detection"]["peak_sign"]
        # matching_params["detect_threshold"] = params["detection"]["detect_threshold"]
        # matching_params["radius_um"] = params["detection"]["radius_um"]

        # spikes = find_spikes_from_templates(
        #     recording, method="tridesclous", method_kwargs=matching_params, **job_kwargs
        # )

        matching_method = params["matching"]["method"]
        matching_params = params["matching"]["method_kwargs"].copy()

        matching_params["templates"] = templates
        matching_params["noise_levels"] = noise_levels
        # matching_params["peak_sign"] = params["detection"]["peak_sign"]
        # matching_params["detect_threshold"] = params["detection"]["detect_threshold"]
        # matching_params["radius_um"] = params["detection"]["radius_um"]

        # spikes = find_spikes_from_templates(
        #     recording, method="tridesclous", method_kwargs=matching_params, **job_kwargs
        # )
        # )

        if matching_method == "circus-omp-svd":
            job_kwargs = job_kwargs.copy()
            for value in ["chunk_size", "chunk_memory", "total_memory", "chunk_duration"]:
                if value in job_kwargs:
                    job_kwargs.pop(value)
            job_kwargs["chunk_duration"] = "100ms"

        spikes = find_spikes_from_templates(
            recording_w, method=matching_method, method_kwargs=matching_params, **job_kwargs
        )

        if params["save_array"]:
            sorting_pre_peeler = sorting_pre_peeler.save(folder=sorter_output_folder / "sorting_pre_peeler")

            np.save(sorter_output_folder / "noise_levels.npy", noise_levels)
            np.save(sorter_output_folder / "all_peaks.npy", all_peaks)
            np.save(sorter_output_folder / "post_split_label.npy", post_split_label)
            np.save(sorter_output_folder / "split_count.npy", split_count)
            np.save(sorter_output_folder / "post_merge_label.npy", post_merge_label)
            np.save(sorter_output_folder / "spikes.npy", spikes)

        final_spikes = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        final_spikes["sample_index"] = spikes["sample_index"]
        final_spikes["unit_index"] = spikes["cluster_index"]
        final_spikes["segment_index"] = spikes["segment_index"]

        sorting = NumpySorting(final_spikes, sampling_frequency, labels_set)
        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        return sorting
