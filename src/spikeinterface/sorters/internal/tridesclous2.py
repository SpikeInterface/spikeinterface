from __future__ import annotations

import shutil
import importlib

from .si_based import ComponentsBasedSorter

from copy import deepcopy

from spikeinterface.core import (
    get_noise_levels,
    NumpySorting,
    estimate_templates_with_accumulator,
    Templates,
    compute_sparsity,
)

from spikeinterface.core.job_tools import fix_job_kwargs

from spikeinterface.preprocessing import bandpass_filter, common_reference, zscore, whiten
from spikeinterface.core.base import minimum_spike_dtype

from spikeinterface.sortingcomponents.tools import cache_preprocessing, clean_cache_preprocessing


import numpy as np


class Tridesclous2Sorter(ComponentsBasedSorter):
    sorter_name = "tridesclous2"

    _default_params = {
        "apply_preprocessing": True,
        "preprocessing_dict": None,
        "apply_motion_correction": False,
        "motion_correction_preset": "dredge_fast",
        "clustering_ms_before": 0.5,
        "clustering_ms_after": 1.5,
        "detection_radius_um": 150.0,
        "features_radius_um": 120.0,
        "split_radius_um" : 60.0,
        "template_radius_um": 100.0,
        "merge_similarity_lag_ms": 0.5,
        "freq_min": 150.0,
        "freq_max": 6000.0,
        "cache_preprocessing_mode": "auto",
        "peak_sign": "neg",
        "detect_threshold": 5.0,
        "n_peaks_per_channel": 5000,
        "n_svd_components_per_channel": 5,
        "n_pca_features": 6,
        "clustering_recursive_depth": 3,
        "ms_before": 2.0,
        "ms_after": 3.0,
        "template_sparsify_threshold": 1.5,
        "template_min_snr_ptp": 3.5,
        "template_max_jitter_ms": 0.2,
        "min_firing_rate": 0.1,
        "gather_mode": "memory",
        "job_kwargs": {},
        "seed": None,
        "save_array": True,
        "debug": False,
    }

    _params_description = {
        "apply_preprocessing": "Apply internal preprocessing or not",
        "preprocessing_dict": "Inject customized preprocessing chain via a dict, instead of the internal one",
        "apply_motion_correction": "Apply motion correction or not (only used when apply_preprocessing=True)",
        "motion_correction_preset": "Motion correction preset",
        "clustering_ms_before": "Milliseconds before the spike peak for clustering",
        "clustering_ms_after": "Milliseconds after the spike peak  for clustering",
        "radius_um": "Radius for sparsity",
        "detection_radius_um": "Radius for peak detection",
        "features_radius_um": "Radius for sparsity in SVD features",
        "split_radius_um" : "Radius for the local split clustering",
        "template_radius_um": "Radius for the sparsity of template before template matching",
        "freq_min": "Low frequency for bandpass filter",
        "freq_max": "High frequency for bandpass filter",
        "peak_sign": "Sign of peaks neg/pos/both",
        "detect_threshold": "Treshold for peak detection",
        "n_peaks_per_channel": "Number of spike per channel for clustering",
        "n_svd_components_per_channel": "Number of SVD components per channel for clustering",
        "n_pca_features": "Secondary PCA features reducation before local isosplit",
        "clustering_recursive_depth": "Clustering recussivity",
        "ms_before": "Milliseconds before the spike peak for template matching",
        "ms_after": "Milliseconds after the spike peak for template matching",
        "template_sparsify_threshold": "Threshold to sparsify templates before template matching",
        "template_min_snr_ptp": "Threshold to remove templates before template matching",
        "template_max_jitter_ms": "Threshold on jitters to remove templates before template matching",
        "min_firing_rate": "To remove small cluster in size before template matching",
        "gather_mode": "How to accumalte spike in matching : memory/npy",
        "job_kwargs": "The famous and fabulous job_kwargs",
        "seed": "Seed for random number",
        "save_array": "Save or not intermediate arrays in the folder",
        "debug": "Save debug files",
    }

    handle_multi_segment = True

    @classmethod
    def get_sorter_version(cls):
        return "2026.01"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):

        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering.main import find_clusters_from_peaks, clustering_methods
        from spikeinterface.preprocessing import correct_motion
        from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
        from spikeinterface.sortingcomponents.tools import clean_templates, compute_sparsity_from_peaks_and_label
        from spikeinterface.preprocessing import apply_preprocessing_pipeline

        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = verbose

        seed = params["seed"]

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        apply_cmr = num_chans >= 32

        # preprocessing
        if params["apply_preprocessing"]:
            if params["apply_motion_correction"]:

                rec_for_motion = recording_raw
                if params["preprocessing_dict"] is None:
                    rec_for_motion = bandpass_filter(
                        rec_for_motion, freq_min=300.0, freq_max=6000.0, ftype="bessel", dtype="float32"
                    )
                    if apply_cmr:
                        rec_for_motion = common_reference(rec_for_motion)
                else:
                    rec_for_motion = apply_preprocessing_pipeline(rec_for_motion, params["preprocessing_dict"])

                if verbose:
                    print("Start correct_motion()")
                _, motion_info = correct_motion(
                    rec_for_motion,
                    folder=sorter_output_folder / "motion",
                    output_motion_info=True,
                    preset=params["motion_correction_preset"],
                )
                if verbose:
                    print("Done correct_motion()")

            if params["preprocessing_dict"] is None:
                recording = bandpass_filter(
                    recording_raw,
                    freq_min=params["freq_min"],
                    freq_max=params["freq_max"],
                    ftype="bessel",
                    filter_order=2,
                    dtype="float32",
                )

                if apply_cmr:
                    recording = common_reference(recording)
            else:
                recording = apply_preprocessing_pipeline(recording_raw, params["preprocessing_dict"])
                recording = recording.astype("float32")

            if params["apply_motion_correction"]:
                interpolate_motion_kwargs = dict(
                    border_mode="force_extrapolate",
                    spatial_interpolation_method="kriging",
                    sigma_um=20.0,
                    p=2,
                )

                recording = InterpolateMotionRecording(
                    recording,
                    motion_info["motion"],
                    **interpolate_motion_kwargs,
                )

            recording = zscore(recording, dtype="float32")
            # whitening is really bad when dirft correction is applied and this changd nothing when no dirft
            # recording = whiten(recording, dtype="float32", mode="local", radius_um=100.0)

            # Cache in mem or folder
            cache_folder = sorter_output_folder / "cache_preprocessing"
            recording_pre_cache = recording
            recording, cache_info = cache_preprocessing(
                recording,
                mode=params["cache_preprocessing_mode"],
                folder=cache_folder,
                job_kwargs=job_kwargs,
            )

            noise_levels = np.ones(num_chans, dtype="float32")
        else:
            recording_pre_cache = recording
            recording = recording_raw.astype("float32")
            noise_levels = get_noise_levels(recording, return_in_uV=False, random_slices_kwargs=dict(seed=seed))
            cache_info = None

        # detection
        detection_params = dict(
            peak_sign=params["peak_sign"],
            detect_threshold=params["detect_threshold"],
            exclude_sweep_ms=1.5,
            radius_um=params["detection_radius_um"],
        )

        all_peaks = detect_peaks(
            recording, method="locally_exclusive", method_kwargs=detection_params, job_kwargs=job_kwargs
        )

        if verbose:
            print(f"detect_peaks(): {len(all_peaks)} peaks found")

        # selection
        n_peaks = max(params["n_peaks_per_channel"] * num_chans, 20_000)
        peaks = select_peaks(all_peaks, method="uniform", n_peaks=n_peaks)

        if verbose:
            print(f"select_peaks(): {len(peaks)} peaks kept for clustering")

        # if clustering_kwargs["clustering"]["clusterer"] == "isosplit6":
        #    have_sisosplit6 = importlib.util.find_spec("isosplit6") is not None
        #    if not have_sisosplit6:
        #        raise ValueError(
        #            "You want to run tridesclous2 with the isosplit6 (the C++) implementation, but this is not installed, please `pip install isosplit6`"
        #        )

        # Clustering
        num_shifts_merging = int(sampling_frequency * params["merge_similarity_lag_ms"] / 1000.)

        clustering_kwargs = deepcopy(clustering_methods["iterative-isosplit"]._default_params)
        clustering_kwargs["peaks_svd"]["ms_before"] = params["clustering_ms_before"]
        clustering_kwargs["peaks_svd"]["ms_after"] = params["clustering_ms_after"]
        clustering_kwargs["peaks_svd"]["radius_um"] = params["features_radius_um"]
        clustering_kwargs["peaks_svd"]["n_components"] = params["n_svd_components_per_channel"]
        clustering_kwargs["split"]["recursive_depth"] = params["clustering_recursive_depth"]
        clustering_kwargs["split"]["method_kwargs"]["n_pca_features"] = params["n_pca_features"]
        clustering_kwargs["clean_templates"]["sparsify_threshold"] = params["template_sparsify_threshold"]
        clustering_kwargs["clean_templates"]["min_snr"] = params["template_min_snr_ptp"]
        clustering_kwargs["clean_templates"]["max_jitter_ms"] = params["template_max_jitter_ms"]
        clustering_kwargs["merge_from_templates"]["num_shifts"] = num_shifts_merging
        clustering_kwargs["noise_levels"] = noise_levels
        clustering_kwargs["clean_low_firing"]["min_firing_rate"] = params["min_firing_rate"]
        clustering_kwargs["clean_low_firing"]["subsampling_factor"] = all_peaks.size / peaks.size
        clustering_kwargs["seed"] = seed

        if params["debug"]:
            clustering_kwargs["debug_folder"] = sorter_output_folder

        unit_ids, clustering_label, more_outs = find_clusters_from_peaks(
            recording,
            peaks,
            method="iterative-isosplit",
            method_kwargs=clustering_kwargs,
            extra_outputs=True,
            job_kwargs=job_kwargs,
        )

        mask = clustering_label >= 0
        kept_peaks = peaks[mask]
        kept_labels = clustering_label[mask]

        sorting_pre_peeler = NumpySorting.from_samples_and_labels(
            kept_peaks["sample_index"],
            kept_labels,
            sampling_frequency,
            unit_ids=unit_ids,
        )
        if verbose:
            print(f"find_clusters_from_peaks(): {unit_ids.size} cluster found")

        recording_for_peeler = recording

        # preestimate the sparsity unsing peaks channel
        spike_vector = sorting_pre_peeler.to_spike_vector(concatenated=True)
        sparsity, unit_locations = compute_sparsity_from_peaks_and_label(
            kept_peaks, spike_vector["unit_index"], sorting_pre_peeler.unit_ids, recording, params["template_radius_um"]
        )

        # we recompute the template even if the clustering give it already because we use different ms_before/ms_after
        ms_before = params["ms_before"]
        ms_after = params["ms_after"]
        nbefore = int(ms_before * sampling_frequency / 1000.0)
        nafter = int(ms_after * sampling_frequency / 1000.0)

        templates_array = estimate_templates_with_accumulator(
            recording_for_peeler,
            sorting_pre_peeler.to_spike_vector(),
            sorting_pre_peeler.unit_ids,
            nbefore,
            nafter,
            return_in_uV=False,
            sparsity_mask=sparsity.mask,
            **job_kwargs,
        )

        templates = Templates(
            templates_array=templates_array,
            sampling_frequency=sampling_frequency,
            nbefore=nbefore,
            channel_ids=recording.channel_ids,
            unit_ids=sorting_pre_peeler.unit_ids,
            sparsity_mask=sparsity.mask,
            probe=recording.get_probe(),
            is_in_uV=False,
        )

        # this clean and sparsify more
        templates = clean_templates(
            templates,
            sparsify_threshold=params["template_sparsify_threshold"],
            noise_levels=noise_levels,
            min_snr=params["template_min_snr_ptp"],
            max_jitter_ms=params["template_max_jitter_ms"],
            remove_empty=True,
        )

        ## peeler
        gather_mode = params["gather_mode"]
        pipeline_kwargs = dict(gather_mode=gather_mode)
        if gather_mode == "npy":
            pipeline_kwargs["folder"] = sorter_output_folder / "matching"
        method_kwargs = dict(noise_levels=noise_levels)
        spikes = find_spikes_from_templates(
            recording_for_peeler,
            templates,
            method="tdc-peeler",
            method_kwargs=method_kwargs,
            pipeline_kwargs=pipeline_kwargs,
            job_kwargs=job_kwargs,
        )

        final_spikes = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        final_spikes["sample_index"] = spikes["sample_index"]
        final_spikes["unit_index"] = spikes["cluster_index"]
        final_spikes["segment_index"] = spikes["segment_index"]
        sorting = NumpySorting(final_spikes, sampling_frequency, templates.unit_ids)

        # auto_merge = True
        auto_merge = False
        analyzer_final = None
        if auto_merge:
            from spikeinterface.sorters.internal.spyking_circus2 import final_cleaning_circus

            analyzer_final = final_cleaning_circus(
                recording_for_peeler,
                sorting,
                templates,
                amplitude_scalings=spikes["amplitude"],
                noise_levels=noise_levels,
                similarity_kwargs={"method": "l1", "support": "union", "max_lag_ms": params["merge_similarity_lag_ms"]},
                sparsity_overlap=0.5,
                censor_ms=3.0,
                max_distance_um=50,
                template_diff_thresh=np.arange(0.05, 0.4, 0.05),
                debug_folder=None,
                job_kwargs=job_kwargs,
            )
            sorting = NumpySorting.from_sorting(analyzer_final.sorting)

        if params["save_array"]:
            sorting_pre_peeler = sorting_pre_peeler.save(folder=sorter_output_folder / "sorting_pre_peeler")

            np.save(sorter_output_folder / "noise_levels.npy", noise_levels)
            np.save(sorter_output_folder / "all_peaks.npy", all_peaks)
            np.save(sorter_output_folder / "peaks.npy", peaks)
            np.save(sorter_output_folder / "clustering_label.npy", clustering_label)
            np.save(sorter_output_folder / "spikes.npy", spikes)
            templates.to_zarr(sorter_output_folder / "templates.zarr")
            if analyzer_final is not None:
                analyzer_final._recording = recording_pre_cache
                analyzer_final.save_as(format="binary_folder", folder=sorter_output_folder / "analyzer")

        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        del recording, recording_for_peeler
        clean_cache_preprocessing(cache_info)

        return sorting
