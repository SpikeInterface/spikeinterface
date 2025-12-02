from __future__ import annotations

from .si_based import ComponentsBasedSorter

import shutil
import numpy as np

from spikeinterface.core import NumpySorting
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.recording_tools import get_noise_levels
from spikeinterface.preprocessing import common_reference, whiten, bandpass_filter, correct_motion
from spikeinterface.sortingcomponents.tools import (
    cache_preprocessing,
    get_shuffled_recording_slices,
    _set_optimal_chunk_size,
)
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core import compute_sparsity


class Spykingcircus2Sorter(ComponentsBasedSorter):
    sorter_name = "spykingcircus2"

    _default_params = {
        "general": {"ms_before": 0.5, "ms_after": 1.5, "radius_um": 100.0},
        "filtering": {"freq_min": 150, "freq_max": 7000, "ftype": "bessel", "filter_order": 2, "margin_ms": 20},
        "whitening": {"mode": "local", "regularize": False},
        "detection": {
            "method": "matched_filtering",
            "method_kwargs": dict(peak_sign="neg", detect_threshold=5),
            "pipeline_kwargs": dict(),
        },
        "selection": {
            "method": "uniform",
            "method_kwargs": dict(n_peaks_per_channel=5000, min_n_peaks=100000, select_per_channel=False),
        },
        "apply_motion_correction": True,
        "motion_correction": {"preset": "dredge_fast"},
        "merging": {"max_distance_um": 50},
        "clustering": {"method": "iterative-hdbscan", "method_kwargs": dict()},
        "cleaning": {"min_snr": 5, "max_jitter_ms": 0.1, "sparsify_threshold": None},
        "matching": {"method": "circus-omp", "method_kwargs": dict(), "pipeline_kwargs": dict()},
        "apply_preprocessing": True,
        "apply_whitening": True,
        "cache_preprocessing": {"mode": "memory", "memory_limit": 0.5, "delete_cache": True},
        "chunk_preprocessing": {"memory_limit": None},
        "multi_units_only": False,
        "job_kwargs": {},
        "seed": 42,
        "deterministic_peaks_detection": False,
        "debug": False,
    }

    handle_multi_segment = True

    _params_description = {
        "general": "A dictionary to describe how templates should be computed. User can define ms_before and ms_after (in ms) \
                                        and also the radius_um used to be considered during clustering",
        "filtering": "A dictionary for the high_pass filter used during preprocessing",
        "whitening": "A dictionary for the whitening used during preprocessing",
        "detection": "A dictionary for the peak detection component. Default is matched-filtering",
        "selection": "A dictionary for the peak selection component. Default is to use uniform",
        "clustering": "A dictionary for the clustering component. Default, iterative-hdbscan is used",
        "matching": "A dictionary for the matching component. Default circus-omp. Use None to avoid matching",
        "merging": "A dictionary to specify the final merging param to group cells after template matching (auto_merge_units)",
        "motion_correction": "A dictionary to be provided if motion correction has to be performed (dense probe only)",
        "apply_preprocessing": "Boolean to specify whether circus 2 should preprocess the recording or not. If yes, then high_pass filtering + common\
                                                    median reference + whitening",
        "apply_motion_correction": "Boolean to specify whether circus 2 should apply motion correction to the recording or not",
        "matched_filtering": "Boolean to specify whether circus 2 should detect peaks via matched filtering (slightly slower)",
        "cache_preprocessing": "How to cache the preprocessed recording. Mode can be memory, file, zarr, with extra arguments. In case of memory (default), \
                         memory_limit will control how much RAM can be used. In case of folder or zarr, delete_cache controls if cache is cleaned after sorting",
        "chunk_preprocessing": "How much RAM (approximately) should be devoted to load all data chunks (given n_jobs).\
                memory_limit will control how much RAM can be used as a fraction of available memory. Otherwise, use total_memory to fix a hard limit, with\
                a string syntax  (e.g. '1G', '500M')",
        "multi_units_only": "Boolean to get only multi units activity (i.e. one template per electrode)",
        "job_kwargs": "A dictionary to specify how many jobs and which parameters they should used",
        "seed": "An int to control how chunks are shuffled while detecting peaks",
        "deterministic_peaks_detection": "A boolean to specify if the peak detection should be deterministic or not. If True, then the seed will be used to shuffle the chunks",
        "debug": "Boolean to specify if internal data structures made during the sorting should be kept for debugging",
    }

    sorter_description = """Spyking Circus 2 is a rewriting of Spyking Circus, within the SpikeInterface framework
    It uses a more conservative clustering algorithm (compared to Spyking Circus), which is less prone to hallucinate units and/or find noise.
    In addition, it also uses a full Orthogonal Matching Pursuit engine to reconstruct the traces, leading to more spikes
    being discovered. The code is much faster and memory efficient, inheriting from all the preprocessing possibilities of spikeinterface"""

    @classmethod
    def get_sorter_version(cls):
        return "2025.10"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):

        try:
            import torch
        except ImportError:
            HAVE_TORCH = False
            print("spykingcircus2 could benefit from using torch. Consider installing it")

        # this is importanted only on demand because numba import are too heavy
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering import find_clusters_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.tools import check_probe_for_drift_correction
        from spikeinterface.sortingcomponents.tools import clean_templates

        job_kwargs = fix_job_kwargs(params["job_kwargs"])
        job_kwargs.update({"progress_bar": verbose})
        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)
        if params["chunk_preprocessing"].get("memory_limit", None) is not None:
            job_kwargs = _set_optimal_chunk_size(recording, job_kwargs, **params["chunk_preprocessing"])

        sampling_frequency = recording.get_sampling_frequency()
        num_channels = recording.get_num_channels()
        ms_before = params["general"].get("ms_before", 0.5)
        ms_after = params["general"].get("ms_after", 1.5)
        radius_um = params["general"].get("radius_um", 100.0)
        detect_threshold = params["detection"]["method_kwargs"].get("detect_threshold", 5)
        peak_sign = params["detection"].get("peak_sign", "neg")
        deterministic = params["deterministic_peaks_detection"]
        debug = params["debug"]
        seed = params["seed"]
        apply_preprocessing = params["apply_preprocessing"]
        apply_whitening = params["apply_whitening"]
        apply_motion_correction = params["apply_motion_correction"]
        exclude_sweep_ms = params["detection"].get("exclude_sweep_ms", max(ms_before, ms_after))

        ## First, we are filtering the data
        filtering_params = params["filtering"].copy()
        if apply_preprocessing:
            if verbose:
                if apply_whitening:
                    print("Preprocessing the recording (bandpass filtering + CMR + whitening)")
                else:
                    print("Preprocessing the recording (bandpass filtering + CMR)")
            recording_f = bandpass_filter(recording, **filtering_params, dtype="float32")
            if num_channels >= 32:
                recording_f = common_reference(recording_f)
        else:
            if verbose:
                print("Skipping preprocessing (whitening only)")
            recording_f = recording
            recording_f.annotate(is_filtered=True)

        if apply_whitening:
            ## We need to whiten before the template matching step, to boost the results
            # TODO add , regularize=True chen ready
            whitening_kwargs = params["whitening"].copy()
            whitening_kwargs["dtype"] = "float32"
            whitening_kwargs["seed"] = params["seed"]
            whitening_kwargs["regularize"] = whitening_kwargs.get("regularize", False)
            if num_channels == 1:
                whitening_kwargs["regularize"] = False
            if whitening_kwargs["regularize"]:
                whitening_kwargs["regularize_kwargs"] = {"method": "LedoitWolf"}
                whitening_kwargs["apply_mean"] = True
            recording_w = whiten(recording_f, **whitening_kwargs)
        else:
            recording_w = recording_f

        valid_geometry = check_probe_for_drift_correction(recording_f)
        if apply_motion_correction:
            if not valid_geometry:
                if verbose:
                    print("Geometry of the probe does not allow 1D drift correction")
                motion_folder = None
            else:
                if verbose:
                    print("Motion correction activated (probe geometry compatible)")
                motion_folder = sorter_output_folder / "motion"
                motion_correction_kwargs = params["motion_correction"].copy()
                motion_correction_kwargs.update({"folder": motion_folder})
                noise_levels = get_noise_levels(
                    recording_w, return_in_uV=False, random_slices_kwargs={"seed": seed}, **job_kwargs
                )
                motion_correction_kwargs["detect_kwargs"] = {"noise_levels": noise_levels}
                recording_w = correct_motion(recording_w, **motion_correction_kwargs, **job_kwargs)
        else:
            motion_folder = None

        noise_levels = get_noise_levels(
            recording_w, return_in_uV=False, random_slices_kwargs={"seed": seed}, **job_kwargs
        )

        if recording_w.check_serializability("json"):
            recording_w.dump(sorter_output_folder / "preprocessed_recording.json", relative_to=None)
        elif recording_w.check_serializability("pickle"):
            recording_w.dump(sorter_output_folder / "preprocessed_recording.pickle", relative_to=None)

        recording_w = cache_preprocessing(recording_w, **job_kwargs, **params["cache_preprocessing"])

        ## Then, we are detecting peaks with a locally_exclusive method
        detection_method = params["detection"].get("method", "matched_filtering")
        detection_params = params["detection"].get("method_kwargs", dict()).copy()
        detection_params["radius_um"] = radius_um / 2
        detection_params["exclude_sweep_ms"] = exclude_sweep_ms
        detect_pipeline_kwargs = params["detection"].get("pipeline_kwargs", dict()).copy()

        matching_method = params["matching"].get("method", "circus-omp")
        matching_params = params["matching"].get("matching_kwargs", dict()).copy()
        matching_pipelines_kwargs = params["matching"].get("pipeline_kwargs", dict())

        clustering_method = params["clustering"].get("method", "iterative-hdbscan")
        clustering_params = params["clustering"].get("method_kwargs", dict()).copy()

        selection_method = params["selection"].get("method", "uniform")
        selection_params = params["selection"].get("method_kwargs", dict()).copy()

        n_peaks_per_channel = selection_params.get("n_peaks_per_channel", 5000)
        min_n_peaks = selection_params.get("min_n_peaks", 100000)
        skip_peaks = not params["multi_units_only"] and selection_method == "uniform"
        skip_peaks = skip_peaks and not deterministic and not (matching_method is None)
        max_n_peaks = n_peaks_per_channel * num_channels
        n_peaks = max(min_n_peaks, max_n_peaks)
        selection_params["n_peaks"] = n_peaks

        if debug:
            clustering_folder = sorter_output_folder / "clustering"
            clustering_folder.mkdir(parents=True, exist_ok=True)
            np.save(clustering_folder / "noise_levels.npy", noise_levels)

        # detection_params["random_chunk_kwargs"] = {"num_chunks_per_segment": 5, "seed": params["seed"]}

        if detection_method == "matched_filtering":
            if not deterministic:
                from spikeinterface.sortingcomponents.tools import (
                    get_prototype_and_waveforms_from_recording,
                )

                detection_params2 = detection_params.copy()
                prototype, waveforms, _ = get_prototype_and_waveforms_from_recording(
                    recording_w,
                    n_peaks=10000,
                    ms_before=ms_before,
                    ms_after=ms_after,
                    seed=seed,
                    noise_levels=noise_levels,
                    job_kwargs=job_kwargs,
                    **detection_params2,
                )
            else:
                from spikeinterface.sortingcomponents.tools import (
                    get_prototype_and_waveforms_from_peaks,
                )

                detection_params2 = detection_params.copy()
                detection_params2["noise_levels"] = noise_levels
                peaks = detect_peaks(
                    recording_w, method="locally_exclusive", method_kwargs=detection_params2, job_kwargs=job_kwargs
                )
                prototype, waveforms, _ = get_prototype_and_waveforms_from_peaks(
                    recording_w,
                    peaks,
                    n_peaks=10000,
                    ms_before=ms_before,
                    ms_after=ms_after,
                    job_kwargs=job_kwargs,
                    seed=seed,
                )
            detection_params["prototype"] = prototype
            detection_params["ms_before"] = ms_before
            if debug:
                np.save(clustering_folder / "waveforms.npy", waveforms)
                np.save(clustering_folder / "prototype.npy", prototype)

        if skip_peaks:
            detect_pipeline_kwargs["recording_slices"] = get_shuffled_recording_slices(
                recording_w,
                job_kwargs=job_kwargs,
                seed=params["seed"],
            )
            detect_pipeline_kwargs["skip_after_n_peaks"] = n_peaks

        peaks = detect_peaks(
            recording_w,
            method=detection_method,
            method_kwargs=detection_params,
            pipeline_kwargs=detect_pipeline_kwargs,
            verbose=verbose,
            job_kwargs=job_kwargs,
        )
        order = np.lexsort((peaks["sample_index"], peaks["segment_index"]))
        peaks = peaks[order]

        if debug:
            np.save(clustering_folder / "peaks.npy", peaks)

        if not skip_peaks and verbose:
            print("Found %d peaks in total" % len(peaks))

        sorting_folder = sorter_output_folder / "sorting"
        if sorting_folder.exists():
            shutil.rmtree(sorting_folder)

        if params["multi_units_only"]:
            sorting = NumpySorting.from_peaks(peaks, sampling_frequency, unit_ids=recording_w.channel_ids)
        else:
            ## We subselect a subset of all the peaks, by making the distributions os SNRs over all
            ## channels as flat as possible
            selected_peaks = select_peaks(peaks, seed=seed, method=selection_method, **selection_params)

            if verbose:
                print("Kept %d peaks for clustering" % len(selected_peaks))

            if clustering_method in [
                "iterative-hdbscan",
                "iterative-isosplit",
                "kilosort-clustering",
                "graph-clustering",
            ]:
                clustering_params.update(verbose=verbose)
                clustering_params.update(seed=seed)
                clustering_params.update(peaks_svd=params["general"])
                if debug:
                    clustering_params["debug_folder"] = sorter_output_folder / "clustering"

            _, peak_labels, more_outs = find_clusters_from_peaks(
                recording_w,
                selected_peaks,
                method=clustering_method,
                method_kwargs=clustering_params,
                extra_outputs=True,
                job_kwargs=job_kwargs,
            )

            clustering_from_svd = True
            for key in ["svd_model", "peaks_svd", "peak_svd_sparse_mask"]:
                if key not in more_outs:
                    clustering_from_svd = False

            if not clustering_from_svd:
                from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_recording

                dense_templates = get_templates_from_peaks_and_recording(
                    recording_w,
                    selected_peaks,
                    peak_labels,
                    ms_before,
                    ms_after,
                    job_kwargs=job_kwargs,
                )

                sparsity = compute_sparsity(dense_templates, method="radius", radius_um=radius_um)
                threshold = params["cleaning"].get("sparsify_threshold", None)
                if threshold is not None:
                    sparsity_snr = compute_sparsity(
                        dense_templates,
                        method="snr",
                        amplitude_mode="peak_to_peak",
                        noise_levels=noise_levels,
                        threshold=threshold,
                    )
                    sparsity.mask = sparsity.mask & sparsity_snr.mask

                templates = dense_templates.to_sparse(sparsity)

            else:
                from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd

                dense_templates, new_sparse_mask = get_templates_from_peaks_and_svd(
                    recording_w,
                    selected_peaks,
                    peak_labels,
                    ms_before,
                    ms_after,
                    more_outs["svd_model"],
                    more_outs["peaks_svd"],
                    more_outs["peak_svd_sparse_mask"],
                    operator="median",
                )
                # this release the peak_svd memmap file
                templates = dense_templates.to_sparse(new_sparse_mask)

            # To be sure that templates have appropriate ms_before and ms_after, up to rounding
            templates.ms_before = ms_before
            templates.ms_after = ms_after

            del more_outs

            cleaning_kwargs = params.get("cleaning", {}).copy()
            cleaning_kwargs["noise_levels"] = noise_levels
            cleaning_kwargs["remove_empty"] = True
            templates = clean_templates(templates, **cleaning_kwargs)

            if verbose:
                print("Kept %d clean clusters" % len(templates.unit_ids))

            if debug:
                templates.to_zarr(folder_path=clustering_folder / "templates")

            ## We launch a OMP matching pursuit by full convolution of the templates and the raw traces

            spikes = find_spikes_from_templates(
                recording_w,
                templates,
                matching_method,
                method_kwargs=matching_params,
                pipeline_kwargs=matching_pipelines_kwargs,
                verbose=verbose,
                job_kwargs=job_kwargs,
            )

            if debug:
                fitting_folder = sorter_output_folder / "fitting"
                fitting_folder.mkdir(parents=True, exist_ok=True)
                np.save(fitting_folder / "spikes", spikes)

            if verbose:
                print("Found %d spikes" % len(spikes))

            ## And this is it! We have a spyking circus
            sorting = np.zeros(spikes.size, dtype=minimum_spike_dtype)
            sorting["sample_index"] = spikes["sample_index"]
            sorting["unit_index"] = spikes["cluster_index"]
            sorting["segment_index"] = spikes["segment_index"]
            sorting = NumpySorting(sorting, sampling_frequency, templates.unit_ids)

            merging_params = params["merging"].copy()
            merging_params["debug_folder"] = sorter_output_folder / "merging"

            if len(merging_params) > 0:
                if params["motion_correction"] and motion_folder is not None:
                    from spikeinterface.preprocessing.motion import load_motion_info

                    motion_info = load_motion_info(motion_folder)
                    motion = motion_info["motion"]
                    max_motion = max(
                        np.max(np.abs(motion.displacement[seg_index])) for seg_index in range(len(motion.displacement))
                    )
                    max_distance_um = merging_params.get("max_distance_um", 50)
                    merging_params["max_distance_um"] = max(max_distance_um, 2 * max_motion)

                if debug:
                    curation_folder = sorter_output_folder / "curation"
                    if curation_folder.exists():
                        shutil.rmtree(curation_folder)
                    sorting.save(folder=curation_folder)
                    # np.save(fitting_folder / "amplitudes", guessed_amplitudes)

                if sorting.get_non_empty_unit_ids().size > 0:
                    final_analyzer = final_cleaning_circus(
                        recording_w,
                        sorting,
                        templates,
                        noise_levels=noise_levels,
                        job_kwargs=job_kwargs,
                        **merging_params,
                    )
                    final_analyzer.save_as(format="binary_folder", folder=sorter_output_folder / "final_analyzer")

                    sorting = final_analyzer.sorting

                if verbose:
                    print(f"Kept {len(sorting.unit_ids)} units after final merging")

        folder_to_delete = None
        cache_mode = params["cache_preprocessing"].get("mode", "memory")
        delete_cache = params["cache_preprocessing"].get("delete_cache", True)

        if cache_mode in ["folder", "zarr"] and delete_cache:
            folder_to_delete = recording_w._kwargs["folder_path"]

        del recording_w
        if folder_to_delete is not None:
            shutil.rmtree(folder_to_delete)

        sorting = sorting.save(folder=sorting_folder)

        return sorting


def final_cleaning_circus(
    recording,
    sorting,
    templates,
    similarity_kwargs={"method": "l1", "support": "union", "max_lag_ms": 0.1},
    sparsity_overlap=0.5,
    censor_ms=3.0,
    max_distance_um=50,
    template_diff_thresh=np.arange(0.05, 0.5, 0.05),
    debug_folder=None,
    noise_levels=None,
    job_kwargs=dict(),
):

    from spikeinterface.sortingcomponents.tools import create_sorting_analyzer_with_existing_templates
    from spikeinterface.curation.auto_merge import auto_merge_units

    # First we compute the needed extensions
    analyzer = create_sorting_analyzer_with_existing_templates(sorting, recording, templates, noise_levels=noise_levels)
    analyzer.compute("unit_locations", method="center_of_mass", **job_kwargs)
    analyzer.compute("template_similarity", **similarity_kwargs)

    if debug_folder is not None:
        analyzer.save_as(format="binary_folder", folder=debug_folder)

    presets = ["x_contaminations"] * len(template_diff_thresh)
    steps_params = [
        {"template_similarity": {"template_diff_thresh": i}, "unit_locations": {"max_distance_um": max_distance_um}}
        for i in template_diff_thresh
    ]
    final_sa = auto_merge_units(
        analyzer,
        presets=presets,
        steps_params=steps_params,
        recursive=True,
        censor_ms=censor_ms,
        sparsity_overlap=sparsity_overlap,
        **job_kwargs,
    )

    return final_sa
