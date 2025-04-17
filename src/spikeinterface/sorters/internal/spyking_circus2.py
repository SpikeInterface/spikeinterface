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
    get_prototype_and_waveforms_from_recording,
    get_shuffled_recording_slices,
)
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.sparsity import compute_sparsity


class Spykingcircus2Sorter(ComponentsBasedSorter):
    sorter_name = "spykingcircus2"

    _default_params = {
        "general": {"ms_before": 2, "ms_after": 2, "radius_um": 100},
        "sparsity": {"method": "snr", "amplitude_mode": "peak_to_peak", "threshold": 0.25},
        "filtering": {"freq_min": 150, "freq_max": 7000, "ftype": "bessel", "filter_order": 2, "margin_ms": 10},
        "whitening": {"mode": "local", "regularize": False},
        "detection": {"method": "matched_filtering", "method_kwargs": dict(peak_sign="neg", detect_threshold=5)},
        "selection": {
            "method": "uniform",
            "method_kwargs": dict(n_peaks_per_channel=5000, min_n_peaks=100000, select_per_channel=False),
        },
        "apply_motion_correction": True,
        "motion_correction": {"preset": "dredge_fast"},
        "merging": {"max_distance_um": 50},
        "clustering": {"method": "circus", "method_kwargs": dict()},
        "matching": {"method": "circus-omp-svd", "method_kwargs": dict()},
        "apply_preprocessing": True,
        "templates_from_svd": True,
        "cache_preprocessing": {"mode": "memory", "memory_limit": 0.5, "delete_cache": True},
        "multi_units_only": False,
        "job_kwargs": {"n_jobs": 0.75},
        "seed": 42,
        "debug": False,
    }

    handle_multi_segment = True

    _params_description = {
        "general": "A dictionary to describe how templates should be computed. User can define ms_before and ms_after (in ms) \
                                        and also the radius_um used to be considered during clustering",
        "sparsity": "A dictionary to be passed to all the calls to sparsify the templates",
        "filtering": "A dictionary for the high_pass filter used during preprocessing",
        "whitening": "A dictionary for the whitening used during preprocessing",
        "detection": "A dictionary for the peak detection component. Default is matched filtering",
        "selection": "A dictionary for the peak selection component. Default is to use uniform",
        "clustering": "A dictionary for the clustering component. Default, graph_clustering is used",
        "matching": "A dictionary for the matching component. Default circus-omp-svd. Use None to avoid matching",
        "merging": "A dictionary to specify the final merging param to group cells after template matching (auto_merge_units)",
        "motion_correction": "A dictionary to be provided if motion correction has to be performed (dense probe only)",
        "apply_preprocessing": "Boolean to specify whether circus 2 should preprocess the recording or not. If yes, then high_pass filtering + common\
                                                    median reference + whitening",
        "apply_motion_correction": "Boolean to specify whether circus 2 should apply motion correction to the recording or not",
        "templates_from_svd": "Boolean to specify whether templates should be computed from SVD or not.",
        "matched_filtering": "Boolean to specify whether circus 2 should detect peaks via matched filtering (slightly slower)",
        "cache_preprocessing": "How to cache the preprocessed recording. Mode can be memory, file, zarr, with extra arguments. In case of memory (default), \
                         memory_limit will control how much RAM can be used. In case of folder or zarr, delete_cache controls if cache is cleaned after sorting",
        "multi_units_only": "Boolean to get only multi units activity (i.e. one template per electrode)",
        "job_kwargs": "A dictionary to specify how many jobs and which parameters they should used",
        "seed": "An int to control how chunks are shuffled while detecting peaks",
        "debug": "Boolean to specify if internal data structures made during the sorting should be kept for debugging",
    }

    sorter_description = """Spyking Circus 2 is a rewriting of Spyking Circus, within the SpikeInterface framework
    It uses a more conservative clustering algorithm (compared to Spyking Circus), which is less prone to hallucinate units and/or find noise.
    In addition, it also uses a full Orthogonal Matching Pursuit engine to reconstruct the traces, leading to more spikes
    being discovered. The code is much faster and memory efficient, inheriting from all the preprocessing possibilities of spikeinterface"""

    @classmethod
    def get_sorter_version(cls):
        return "2.1"

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
        from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.tools import remove_empty_templates
        from spikeinterface.sortingcomponents.tools import check_probe_for_drift_correction

        job_kwargs = fix_job_kwargs(params["job_kwargs"])
        job_kwargs.update({"progress_bar": verbose})

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        sampling_frequency = recording.get_sampling_frequency()
        num_channels = recording.get_num_channels()
        ms_before = params["general"].get("ms_before", 2)
        ms_after = params["general"].get("ms_after", 2)
        radius_um = params["general"].get("radius_um", 75)
        peak_sign = params["detection"].get("peak_sign", "neg")
        templates_from_svd = params["templates_from_svd"]
        debug = params["debug"]
        seed = params["seed"]
        apply_preprocessing = params["apply_preprocessing"]
        apply_motion_correction = params["apply_motion_correction"]
        exclude_sweep_ms = params["detection"].get("exclude_sweep_ms", max(ms_before, ms_after))

        ## First, we are filtering the data
        filtering_params = params["filtering"].copy()
        if apply_preprocessing:
            if verbose:
                print("Preprocessing the recording (bandpass filtering + CMR + whitening)")
            recording_f = bandpass_filter(recording, **filtering_params, dtype="float32")
            if num_channels > 1:
                recording_f = common_reference(recording_f)
        else:
            if verbose:
                print("Skipping preprocessing (whitening only)")
            recording_f = recording
            recording_f.annotate(is_filtered=True)

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
                params["motion_correction"].update({"folder": motion_folder})
                recording_f = correct_motion(recording_f, **params["motion_correction"], **job_kwargs)
        else:
            motion_folder = None

        ## We need to whiten before the template matching step, to boost the results
        # TODO add , regularize=True chen ready
        whitening_kwargs = params["whitening"].copy()
        whitening_kwargs["dtype"] = "float32"
        whitening_kwargs["regularize"] = whitening_kwargs.get("regularize", False)
        if num_channels == 1:
            whitening_kwargs["regularize"] = False
        if whitening_kwargs["regularize"]:
            whitening_kwargs["regularize_kwargs"] = {"method": "LedoitWolf"}
            whitening_kwargs["apply_mean"] = True

        recording_w = whiten(recording_f, **whitening_kwargs)

        noise_levels = get_noise_levels(recording_w, return_scaled=False, **job_kwargs)

        if recording_w.check_serializability("json"):
            recording_w.dump(sorter_output_folder / "preprocessed_recording.json", relative_to=None)
        elif recording_w.check_serializability("pickle"):
            recording_w.dump(sorter_output_folder / "preprocessed_recording.pickle", relative_to=None)

        recording_w = cache_preprocessing(recording_w, **job_kwargs, **params["cache_preprocessing"])

        ## Then, we are detecting peaks with a locally_exclusive method
        detection_method = params["detection"].get("method", "matched_filtering")
        detection_params = params["detection"].get("method_kwargs", dict())
        detection_params["radius_um"] = radius_um
        detection_params["exclude_sweep_ms"] = exclude_sweep_ms
        detection_params["noise_levels"] = noise_levels

        selection_method = params["selection"].get("method", "uniform")
        selection_params = params["selection"].get("method_kwargs", dict())
        n_peaks_per_channel = selection_params.get("n_peaks_per_channel", 5000)
        min_n_peaks = selection_params.get("min_n_peaks", 100000)
        skip_peaks = not params["multi_units_only"] and selection_method == "uniform"
        max_n_peaks = n_peaks_per_channel * num_channels
        n_peaks = max(min_n_peaks, max_n_peaks)
        selection_params["n_peaks"] = n_peaks
        selection_params["noise_levels"] = noise_levels

        if debug:
            clustering_folder = sorter_output_folder / "clustering"
            clustering_folder.mkdir(parents=True, exist_ok=True)
            np.save(clustering_folder / "noise_levels.npy", noise_levels)

        if detection_method == "matched_filtering":
            prototype, waveforms, _ = get_prototype_and_waveforms_from_recording(
                recording_w,
                n_peaks=10000,
                ms_before=ms_before,
                ms_after=ms_after,
                seed=seed,
                **detection_params,
                **job_kwargs,
            )
            detection_params["prototype"] = prototype
            detection_params["ms_before"] = ms_before
            if debug:
                np.save(clustering_folder / "waveforms.npy", waveforms)
                np.save(clustering_folder / "prototype.npy", prototype)
            if skip_peaks:
                detection_params["skip_after_n_peaks"] = n_peaks
                detection_params["recording_slices"] = get_shuffled_recording_slices(
                    recording_w, seed=seed, **job_kwargs
                )
            detection_method = "matched_filtering"
        else:
            waveforms = None
            if skip_peaks:
                detection_params["skip_after_n_peaks"] = n_peaks
                detection_params["recording_slices"] = get_shuffled_recording_slices(
                    recording_w, seed=seed, **job_kwargs
                )
            detection_method = "locally_exclusive"

        peaks = detect_peaks(recording_w, detection_method, **detection_params, **job_kwargs)
        order = np.lexsort((peaks["sample_index"], peaks["segment_index"]))
        peaks = peaks[order]

        if debug:
            np.save(clustering_folder / "peaks.npy", peaks)

        if not skip_peaks and verbose:
            print("Found %d peaks in total" % len(peaks))

        sparsity_kwargs = params["sparsity"].copy()
        if "peak_sign" not in sparsity_kwargs:
            sparsity_kwargs["peak_sign"] = peak_sign

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

            clustering_method = params["clustering"].get("method", "graph_clustering")
            clustering_params = params["clustering"].get("method_kwargs", dict())

            if clustering_method == "circus":
                clustering_params["waveforms"] = {}
                clustering_params["sparsity"] = sparsity_kwargs
                clustering_params["neighbors_radius_um"] = 50
                clustering_params["radius_um"] = radius_um
                clustering_params["waveforms"]["ms_before"] = ms_before
                clustering_params["waveforms"]["ms_after"] = ms_after
                clustering_params["few_waveforms"] = waveforms
                clustering_params["noise_levels"] = noise_levels
                clustering_params["ms_before"] = ms_before
                clustering_params["ms_after"] = ms_after
                clustering_params["verbose"] = verbose
                clustering_params["templates_from_svd"] = templates_from_svd
                clustering_params["tmp_folder"] = sorter_output_folder / "clustering"
                clustering_params["debug"] = debug
                clustering_params["noise_threshold"] = detection_params.get("detect_threshold", 4)
            elif clustering_method == "graph_clustering":
                clustering_params = {
                    "ms_before": ms_before,
                    "ms_after": ms_after,
                    "clustering_method": "hdbscan",
                    "radius_um": radius_um,
                    "clustering_kwargs": dict(
                        min_samples=1,
                        min_cluster_size=50,
                        core_dist_n_jobs=-1,
                        cluster_selection_method="leaf",
                        allow_single_cluster=True,
                        cluster_selection_epsilon=0.1,
                    ),
                }

            outputs = find_cluster_from_peaks(
                recording_w,
                selected_peaks,
                method=clustering_method,
                method_kwargs=clustering_params,
                extra_outputs=templates_from_svd,
                **job_kwargs,
            )

            if len(outputs) == 2:
                _, peak_labels = outputs
                from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_recording

                templates = get_templates_from_peaks_and_recording(
                    recording_w,
                    selected_peaks,
                    peak_labels,
                    ms_before,
                    ms_after,
                    **job_kwargs,
                )
            elif len(outputs) == 5:
                _, peak_labels, svd_model, svd_features, sparsity_mask = outputs
                from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd

                templates = get_templates_from_peaks_and_svd(
                    recording_w,
                    selected_peaks,
                    peak_labels,
                    ms_before,
                    ms_after,
                    svd_model,
                    svd_features,
                    sparsity_mask,
                    operator="median",
                )

            sparsity = compute_sparsity(templates, noise_levels, **sparsity_kwargs)
            templates = templates.to_sparse(sparsity)
            templates = remove_empty_templates(templates)

            if debug:
                templates.to_zarr(folder_path=clustering_folder / "templates")

            ## We launch a OMP matching pursuit by full convolution of the templates and the raw traces
            matching_method = params["matching"].get("method", "circus-omp_svd")
            matching_params = params["matching"].get("method_kwargs", dict())
            matching_params["templates"] = templates

            if matching_method is not None:
                spikes = find_spikes_from_templates(
                    recording_w, matching_method, method_kwargs=matching_params, **job_kwargs
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
            else:
                ## we should have a case to deal with clustering all peaks without matching
                ## for small density channel counts

                sorting = np.zeros(selected_peaks.size, dtype=minimum_spike_dtype)
                sorting["sample_index"] = selected_peaks["sample_index"]
                sorting["unit_index"] = peak_labels
                sorting["segment_index"] = selected_peaks["segment_index"]
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

                sorting = final_cleaning_circus(recording_w, sorting, templates, **merging_params, **job_kwargs)

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
    similarity_kwargs={"method": "l2", "support": "union", "max_lag_ms": 0.1},
    sparsity_overlap=0.5,
    censor_ms=3.0,
    max_distance_um=50,
    template_diff_thresh=np.arange(0.05, 0.5, 0.05),
    debug_folder=None,
    **job_kwargs,
):

    from spikeinterface.sortingcomponents.tools import create_sorting_analyzer_with_existing_templates
    from spikeinterface.curation.auto_merge import auto_merge_units

    # First we compute the needed extensions
    analyzer = create_sorting_analyzer_with_existing_templates(sorting, recording, templates)
    analyzer.compute("unit_locations", method="monopolar_triangulation")
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
    return final_sa.sorting
