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
from spikeinterface.core.basesorting import minimum_spike_dtype

from spikeinterface.sortingcomponents.tools import cache_preprocessing, clean_cache_preprocessing


import numpy as np


class Tridesclous2Sorter(ComponentsBasedSorter):
    sorter_name = "tridesclous2"

    _default_params = {
        "apply_preprocessing": True,
        "apply_motion_correction": False,
        "motion_correction": {"preset": "dredge_fast"},
        "cache_preprocessing_mode": "auto",
        "waveforms": {
            "ms_before": 0.5,
            "ms_after": 1.5,
            "radius_um": 120.0,
        },
        "filtering": {
            "freq_min": 150.0,
            "freq_max": 6000.0,
            "ftype": "bessel",
            "filter_order": 2,
        },
        "detection": {"peak_sign": "neg", "detect_threshold": 5, "exclude_sweep_ms": 1.5, "radius_um": 150.0},
        "selection": {"n_peaks_per_channel": 5000, "min_n_peaks": 20000},
        "svd": {"n_components": 5},
        "clustering": {
            "recursive_depth": 3,
        },
        "templates": {
            "ms_before": 2.0,
            "ms_after": 3.0,
            "max_spikes_per_unit": 400,
            "sparsity_threshold": 1.5,
            "min_snr": 2.5,
            "radius_um": 100.0,
            "max_jitter_ms": 0.2,
        },
        "matching": {"method": "tdc-peeler", "method_kwargs": {}, "gather_mode": "memory"},
        "job_kwargs": {},
        "save_array": True,
        "debug": False,
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
        return "2025.11"

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):

        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering.main import find_clusters_from_peaks, clustering_methods
        from spikeinterface.preprocessing import correct_motion
        from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
        from spikeinterface.sortingcomponents.tools import clean_templates, compute_sparsity_from_peaks_and_label

        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = verbose

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        apply_cmr = num_chans >= 32

        # preprocessing
        if params["apply_preprocessing"]:
            if params["apply_motion_correction"]:
                rec_for_motion = recording_raw
                if params["apply_preprocessing"]:
                    rec_for_motion = bandpass_filter(
                        rec_for_motion, freq_min=300.0, freq_max=6000.0, ftype="bessel", dtype="float32"
                    )
                    if apply_cmr:
                        rec_for_motion = common_reference(rec_for_motion)
                    if verbose:
                        print("Start correct_motion()")
                    _, motion_info = correct_motion(
                        rec_for_motion,
                        folder=sorter_output_folder / "motion",
                        output_motion_info=True,
                        **params["motion_correction"],
                    )
                    if verbose:
                        print("Done correct_motion()")

            recording = bandpass_filter(recording_raw, **params["filtering"], margin_ms=20.0, dtype="float32")

            if apply_cmr:
                recording = common_reference(recording)

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
            recording, cache_info = cache_preprocessing(
                recording,
                mode=params["cache_preprocessing_mode"],
                folder=cache_folder,
                job_kwargs=job_kwargs,
            )

            noise_levels = np.ones(num_chans, dtype="float32")
        else:
            recording = recording_raw
            noise_levels = get_noise_levels(recording, return_in_uV=False)
            cache_info = None

        # detection
        detection_params = params["detection"].copy()
        detection_params["noise_levels"] = noise_levels
        all_peaks = detect_peaks(
            recording, method="locally_exclusive", method_kwargs=detection_params, job_kwargs=job_kwargs
        )

        if verbose:
            print(f"detect_peaks(): {len(all_peaks)} peaks found")

        # selection
        selection_params = params["selection"].copy()
        n_peaks = params["selection"]["n_peaks_per_channel"] * num_chans
        n_peaks = max(selection_params["min_n_peaks"], n_peaks)
        peaks = select_peaks(all_peaks, method="uniform", n_peaks=n_peaks)

        if verbose:
            print(f"select_peaks(): {len(peaks)} peaks kept for clustering")

        # routing clustering params into the big IterativeISOSPLITClustering params tree
        clustering_kwargs = deepcopy(clustering_methods["iterative-isosplit"]._default_params)
        clustering_kwargs["peaks_svd"].update(params["waveforms"])
        clustering_kwargs["peaks_svd"].update(params["svd"])
        clustering_kwargs["split"].update(params["clustering"])
        if params["debug"]:
            clustering_kwargs["debug_folder"] = sorter_output_folder
        clustering_kwargs["noise_levels"] = noise_levels

        # if clustering_kwargs["clustering"]["clusterer"] == "isosplit6":
        #    have_sisosplit6 = importlib.util.find_spec("isosplit6") is not None
        #    if not have_sisosplit6:
        #        raise ValueError(
        #            "You want to run tridesclous2 with the isosplit6 (the C++) implementation, but this is not installed, please `pip install isosplit6`"
        #        )

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
            kept_peaks,
            spike_vector["unit_index"],
            sorting_pre_peeler.unit_ids,
            recording,
            params["templates"]["radius_um"],
        )

        # we recompute the template even if the clustering give it already because we use different ms_before/ms_after
        nbefore = int(params["templates"]["ms_before"] * sampling_frequency / 1000.0)
        nafter = int(params["templates"]["ms_after"] * sampling_frequency / 1000.0)

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

        # this spasify more
        templates = clean_templates(
            templates,
            sparsify_threshold=params["templates"]["sparsity_threshold"],
            noise_levels=noise_levels,
            min_snr=params["templates"]["min_snr"],
            max_jitter_ms=params["templates"]["max_jitter_ms"],
            remove_empty=True,
        )

        ## peeler
        matching_method = params["matching"].pop("method")
        gather_mode = params["matching"].pop("gather_mode", "memory")
        matching_params = params["matching"].get("matching_kwargs", {}).copy()
        if matching_method in ("tdc-peeler",):
            matching_params["noise_levels"] = noise_levels

        pipeline_kwargs = dict(gather_mode=gather_mode)
        if gather_mode == "npy":
            pipeline_kwargs["folder"] = sorter_output_folder / "matching"
        spikes = find_spikes_from_templates(
            recording_for_peeler,
            templates,
            method=matching_method,
            method_kwargs=matching_params,
            pipeline_kwargs=pipeline_kwargs,
            job_kwargs=job_kwargs,
        )

        final_spikes = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        final_spikes["sample_index"] = spikes["sample_index"]
        final_spikes["unit_index"] = spikes["cluster_index"]
        final_spikes["segment_index"] = spikes["segment_index"]
        sorting = NumpySorting(final_spikes, sampling_frequency, templates.unit_ids)

        auto_merge = True
        analyzer_final = None
        if auto_merge:
            from spikeinterface.sorters.internal.spyking_circus2 import final_cleaning_circus

            # max_distance_um = merging_params.get("max_distance_um", 50)
            # merging_params["max_distance_um"] = max(max_distance_um, 2 * max_motion)

            analyzer_final = final_cleaning_circus(
                recording_for_peeler,
                sorting,
                templates,
                similarity_kwargs={"method": "l1", "support": "union", "max_lag_ms": 0.1},
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
                analyzer_final.save_as(format="binary_folder", folder=sorter_output_folder / "analyzer")

        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        del recording, recording_for_peeler
        clean_cache_preprocessing(cache_info)

        return sorting
