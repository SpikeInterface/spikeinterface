from __future__ import annotations

import shutil

from .si_based import ComponentsBasedSorter

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

from spikeinterface.sortingcomponents.tools import cache_preprocessing


import numpy as np


class Tridesclous2Sorter(ComponentsBasedSorter):
    sorter_name = "tridesclous2"

    _default_params = {
        "apply_preprocessing": True,
        "apply_motion_correction": False,
        "motion_correction": {"preset": "nonrigid_fast_and_accurate"},
        "cache_preprocessing": {"mode": "memory", "memory_limit": 0.5, "delete_cache": True},
        "waveforms": {
            "ms_before": 0.5,
            "ms_after": 1.5,
            "radius_um": 120.0,
        },
        "filtering": {"freq_min": 300.0, "freq_max": 8000.0},
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
            "sparsity_threshold": 1.5,
            # "peak_shift_ms": 0.2,
        },
        # "matching": {"method": "tridesclous", "method_kwargs": {"peak_shift_ms": 0.2, "radius_um": 100.0}},
        # "matching": {"method": "circus-omp-svd", "method_kwargs": {}},
        "matching": {"method": "wobble", "method_kwargs": {}},
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

        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.peak_detection import detect_peaks
        from spikeinterface.sortingcomponents.peak_selection import select_peaks
        from spikeinterface.sortingcomponents.clustering.main import find_cluster_from_peaks
        from spikeinterface.sortingcomponents.tools import remove_empty_templates
        from spikeinterface.preprocessing import correct_motion
        from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

        job_kwargs = params["job_kwargs"].copy()
        job_kwargs = fix_job_kwargs(job_kwargs)
        job_kwargs["progress_bar"] = verbose

        recording_raw = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        num_chans = recording_raw.get_num_channels()
        sampling_frequency = recording_raw.get_sampling_frequency()

        # preprocessing
        if params["apply_preprocessing"]:
            if params["apply_motion_correction"]:
                rec_for_motion = recording_raw
                if params["apply_preprocessing"]:
                    rec_for_motion = bandpass_filter(rec_for_motion, freq_min=300.0, freq_max=6000.0, dtype="float32")
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

            recording = bandpass_filter(recording_raw, **params["filtering"], dtype="float32")
            recording = common_reference(recording)

            if params["apply_motion_correction"]:
                interpolate_motion_kwargs = dict(
                    direction=1,
                    border_mode="force_extrapolate",
                    spatial_interpolation_method="kriging",
                    sigma_um=20.0,
                    p=2,
                )

                recording = InterpolateMotionRecording(
                    recording,
                    motion_info["motion"],
                    motion_info["temporal_bins"],
                    motion_info["spatial_bins"],
                    **interpolate_motion_kwargs,
                )

            recording = zscore(recording, dtype="float32")
            recording = whiten(recording, dtype="float32", mode="local", radius_um=100.0)

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
            print(f"detect_peaks(): {len(all_peaks)} peaks found")

        # selection
        selection_params = params["selection"].copy()
        n_peaks = params["selection"]["n_peaks_per_channel"] * num_chans
        n_peaks = max(selection_params["min_n_peaks"], n_peaks)
        peaks = select_peaks(all_peaks, method="uniform", n_peaks=n_peaks)

        if verbose:
            print(f"select_peaks(): {len(peaks)} peaks kept for clustering")

        clustering_kwargs = {}
        clustering_kwargs["folder"] = sorter_output_folder
        clustering_kwargs["waveforms"] = params["waveforms"].copy()
        clustering_kwargs["clustering"] = params["clustering"].copy()

        labels_set, clustering_label, extra_out = find_cluster_from_peaks(
            recording, peaks, method="tdc_clustering", method_kwargs=clustering_kwargs, extra_outputs=True, **job_kwargs
        )
        peak_shifts = extra_out["peak_shifts"]
        new_peaks = peaks.copy()
        new_peaks["sample_index"] -= peak_shifts

        mask = clustering_label >= 0
        sorting_pre_peeler = NumpySorting.from_times_labels(
            new_peaks["sample_index"][mask],
            clustering_label[mask],
            sampling_frequency,
            unit_ids=labels_set,
        )

        if verbose:
            print(f"find_cluster_from_peaks(): {sorting_pre_peeler.unit_ids.size} cluster found")

        recording_for_peeler = recording

        nbefore = int(params["templates"]["ms_before"] * sampling_frequency / 1000.0)
        nafter = int(params["templates"]["ms_after"] * sampling_frequency / 1000.0)

        templates_array = estimate_templates_with_accumulator(
            recording_for_peeler,
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
            sparsity_mask=None,
            probe=recording_for_peeler.get_probe(),
            is_scaled=False,
        )

        # TODO : try other methods for sparsity
        sparsity_threshold = params["templates"]["sparsity_threshold"]
        # sparsity = compute_sparsity(templates_dense, method="radius", radius_um=120.)
        sparsity = compute_sparsity(templates_dense, noise_levels=noise_levels, threshold=sparsity_threshold)
        templates = templates_dense.to_sparse(sparsity)
        templates = remove_empty_templates(templates)

        ## peeler
        matching_method = params["matching"]["method"]
        matching_params = params["matching"]["method_kwargs"].copy()
        matching_params["templates"] = templates
        matching_params["noise_levels"] = noise_levels
        spikes = find_spikes_from_templates(
            recording_for_peeler, method=matching_method, method_kwargs=matching_params, **job_kwargs
        )

        if params["save_array"]:
            sorting_pre_peeler = sorting_pre_peeler.save(folder=sorter_output_folder / "sorting_pre_peeler")

            np.save(sorter_output_folder / "noise_levels.npy", noise_levels)
            np.save(sorter_output_folder / "all_peaks.npy", all_peaks)
            np.save(sorter_output_folder / "peaks.npy", peaks)
            np.save(sorter_output_folder / "clustering_label.npy", clustering_label)
            np.save(sorter_output_folder / "spikes.npy", spikes)

        final_spikes = np.zeros(spikes.size, dtype=minimum_spike_dtype)
        final_spikes["sample_index"] = spikes["sample_index"]
        final_spikes["unit_index"] = spikes["cluster_index"]
        final_spikes["segment_index"] = spikes["segment_index"]

        sorting = NumpySorting(final_spikes, sampling_frequency, labels_set)
        sorting = sorting.save(folder=sorter_output_folder / "sorting")

        return sorting
