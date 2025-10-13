from __future__ import annotations

from collections import namedtuple
import numpy as np
from spikeinterface.core.analyzer_extension_core import BaseMetric
from spikeinterface.metrics.quality.misc_metrics_implementations import (
    compute_noise_cutoffs,
    compute_num_spikes,
    compute_firing_rates,
    compute_presence_ratios,
    compute_snrs,
    compute_isi_violations,
    compute_refrac_period_violations,
    compute_sliding_rp_violations,
    compute_synchrony_metrics,
    compute_firing_ranges,
    compute_amplitude_cv_metrics,
    compute_amplitude_cutoffs,
    compute_amplitude_medians,
    compute_drift_metrics,
    compute_sd_ratio,
)
from spikeinterface.metrics.quality.pca_metrics_implementations import (
    mahalanobis_metrics,
    lda_metrics,
    nearest_neighbors_metrics,
    nearest_neighbors_isolation,
    nearest_neighbors_noise_overlap,
    simplified_silhouette_score,
    silhouette_score,
)
from spikeinterface.core.template_tools import get_template_extremum_channel


# # TODO: move to spiketrain metrics
# def _num_spikes_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
#     num_spikes_result = namedtuple("NumSpikesResult", ["num_spikes"])
#     result = compute_num_spikes(sorting_analyzer, unit_ids=unit_ids, **metric_params)
#     return num_spikes_result(num_spikes=result)


class NumSpikes(BaseMetric):
    metric_name = "num_spikes"
    metric_function = compute_num_spikes
    metric_params = {}
    metric_columns = ["num_spikes"]
    metric_dtypes = {"num_spikes": int}


# def _firing_rate_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
#     firing_rate_result = namedtuple("FiringRateResult", ["firing_rate"])
#     result = compute_firing_rates(sorting_analyzer, unit_ids=unit_ids)
#     return firing_rate_result(firing_rate=result)


class FiringRate(BaseMetric):
    metric_name = "firing_rate"
    metric_function = compute_firing_rates
    metric_params = {}
    metric_columns = ["firing_rate"]
    metric_dtypes = {"firing_rate": float}


def _noise_cutoff_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    noise_cutoff_result = namedtuple("NoiseCutoffResult", ["noise_cutoff", "noise_ratio"])
    result = compute_noise_cutoffs(sorting_analyzer, unit_ids=unit_ids, **metric_params)
    return noise_cutoff_result(noise_cutoff=result.noise_cutoff, noise_ratio=result.noise_ratio)


class NoiseCutoff(BaseMetric):
    metric_name = "noise_cutoff"
    metric_function = _noise_cutoff_metric_function
    metric_params = {"high_quantile": 0.25, "low_quantile": 0.1, "n_bins": 100}
    metric_columns = ["noise_cutoff", "noise_ratio"]
    metric_dtypes = {"noise_cutoff": float, "noise_ratio": float}


class PresenceRatio(BaseMetric):
    metric_name = "presence_ratio"
    metric_function = compute_presence_ratios
    metric_params = {"bin_duration_s": 60, "mean_fr_ratio_thresh": 0.0}
    metric_columns = ["presence_ratio"]
    metric_dtypes = {"presence_ratio": float}


class SNR(BaseMetric):
    metric_name = "snr"
    metric_function = compute_snrs
    metric_params = {"peak_sign": "neg", "peak_mode": "extremum"}
    metric_columns = ["snr"]
    metric_dtypes = {"snr": float}
    depend_on = ["noise_levels", "templates"]


class ISIViolation(BaseMetric):
    metric_name = "isi_violation"
    metric_function = compute_isi_violations
    metric_params = {"isi_threshold_ms": 1.5, "min_isi_ms": 0}
    metric_columns = ["isi_violations_ratio", "isi_violations_count"]
    metric_dtypes = {"isi_violations_ratio": float, "isi_violations_count": int}


class RPViolation(BaseMetric):
    metric_name = "rp_violation"
    metric_function = compute_refrac_period_violations
    metric_params = {"refractory_period_ms": 1.0, "censored_period_ms": 0.0}
    metric_columns = ["rp_contamination", "rp_violations"]
    metric_dtypes = {"rp_contamination": float, "rp_violations": int}


class SlidingRPViolation(BaseMetric):
    metric_name = "sliding_rp_violation"
    metric_function = compute_sliding_rp_violations
    metric_params = {
        "min_spikes": 0,
        "bin_size_ms": 0.25,
        "window_size_s": 1,
        "exclude_ref_period_below_ms": 0.5,
        "max_ref_period_ms": 10,
        "contamination_values": None,
    }
    metric_columns = ["sliding_rp_violation"]
    metric_dtypes = {"sliding_rp_violation": float}


class Synchrony(BaseMetric):
    metric_name = "synchrony"
    metric_function = compute_synchrony_metrics
    metric_params = {}
    metric_columns = ["sync_spike_2", "sync_spike_4", "sync_spike_8"]
    metric_dtypes = {"sync_spike_2": float, "sync_spike_4": float, "sync_spike_8": float}


class FiringRange(BaseMetric):
    metric_name = "firing_range"
    metric_function = compute_firing_ranges
    metric_params = {"bin_size_s": 5, "percentiles": (5, 95)}
    metric_columns = ["firing_range"]
    metric_dtypes = {"firing_range": float}


class AmplitudeCV(BaseMetric):
    metric_name = "amplitude_cv"
    metric_function = compute_amplitude_cv_metrics
    metric_params = {
        "average_num_spikes_per_bin": 50,
        "percentiles": (5, 95),
        "min_num_bins": 10,
        "amplitude_extension": "spike_amplitudes",
    }
    metric_columns = ["amplitude_cv_median", "amplitude_cv_range"]
    metric_dtypes = {"amplitude_cv_median": float, "amplitude_cv_range": float}
    depend_on = ["spike_amplitudes|amplitude_scalings"]


class AmplitudeCutoff(BaseMetric):
    metric_name = "amplitude_cutoff"
    metric_function = compute_amplitude_cutoffs
    metric_params = {
        "peak_sign": "neg",
        "num_histogram_bins": 100,
        "histogram_smoothing_value": 3,
        "amplitudes_bins_min_ratio": 5,
    }
    metric_columns = ["amplitude_cutoff"]
    metric_dtypes = {"amplitude_cutoff": float}
    depend_on = ["spike_amplitudes|amplitude_scalings"]


class AmplitudeMedian(BaseMetric):
    metric_name = "amplitude_median"
    metric_function = compute_amplitude_medians
    metric_params = {"peak_sign": "neg"}
    metric_columns = ["amplitude_median"]
    metric_dtypes = {"amplitude_median": float}
    depend_on = ["spike_amplitudes"]


class Drift(BaseMetric):
    metric_name = "drift"
    metric_function = compute_drift_metrics
    metric_params = {
        "interval_s": 60,
        "min_spikes_per_interval": 100,
        "direction": "y",
        "min_num_bins": 2,
    }
    metric_columns = ["drift_ptp", "drift_std", "drift_mad"]
    metric_dtypes = {"drift_ptp": float, "drift_std": float, "drift_mad": float}
    depend_on = ["spike_locations"]


def _sd_ratio_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    sd_ratio_result = namedtuple("SDRatioResult", ["sd_ratio"])
    result = compute_sd_ratio(sorting_analyzer, unit_ids=unit_ids, **metric_params)
    return sd_ratio_result(sd_ratio=result)


class SDRatio(BaseMetric):
    metric_name = "sd_ratio"
    metric_function = _sd_ratio_metric_function
    metric_params = {
        "censored_period_ms": 4.0,
        "correct_for_drift": True,
        "correct_for_template_itself": True,
    }
    metric_columns = ["sd_ratio"]
    metric_dtypes = {"sd_ratio": float}
    needs_recording = True
    depend_on = ["templates", "spike_amplitudes"]


# Group metrics into categories
misc_metrics = [
    NoiseCutoff,
    NumSpikes,
    FiringRate,
    PresenceRatio,
    SNR,
    ISIViolation,
    RPViolation,
    SlidingRPViolation,
    Synchrony,
    FiringRange,
    AmplitudeCV,
    AmplitudeCutoff,
    AmplitudeMedian,
    Drift,
    SDRatio,
]

# PCA-based metrics


def _mahalanobis_metrics_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    mahalanobis_result = namedtuple("MahalanobisResult", ["isolation_distance", "l_ratio"])

    # Use pre-computed PCA data
    pca_data_per_unit = tmp_data["pca_data_per_unit"]

    isolation_distance_dict = {}
    l_ratio_dict = {}

    for unit_id in unit_ids:
        pcs_flat = pca_data_per_unit[unit_id]["pcs_flat"]
        labels = pca_data_per_unit[unit_id]["labels"]

        try:
            isolation_distance, l_ratio = mahalanobis_metrics(pcs_flat, labels, unit_id)
        except:
            isolation_distance = np.nan
            l_ratio = np.nan

        isolation_distance_dict[unit_id] = isolation_distance
        l_ratio_dict[unit_id] = l_ratio

    return mahalanobis_result(isolation_distance=isolation_distance_dict, l_ratio=l_ratio_dict)


class MahalanobisMetrics(BaseMetric):
    metric_name = "mahalanobis_metrics"
    metric_function = _mahalanobis_metrics_function
    metric_params = {}
    metric_columns = ["isolation_distance", "l_ratio"]
    metric_dtypes = {"isolation_distance": float, "l_ratio": float}
    depend_on = ["principal_components"]


def _d_prime_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    d_prime_result = namedtuple("DPrimeResult", ["d_prime"])

    # Use pre-computed PCA data
    pca_data_per_unit = tmp_data["pca_data_per_unit"]

    d_prime_dict = {}

    for unit_id in unit_ids:
        if len(unit_ids) == 1:
            d_prime_dict[unit_id] = np.nan
            continue

        pcs_flat = pca_data_per_unit[unit_id]["pcs_flat"]
        labels = pca_data_per_unit[unit_id]["labels"]

        try:
            d_prime = lda_metrics(pcs_flat, labels, unit_id)
        except:
            d_prime = np.nan

        d_prime_dict[unit_id] = d_prime

    return d_prime_result(d_prime=d_prime_dict)


class DPrimeMetrics(BaseMetric):
    metric_name = "d_prime"
    metric_function = _d_prime_metric_function
    metric_params = {}
    metric_columns = ["d_prime"]
    metric_dtypes = {"d_prime": float}
    depend_on = ["principal_components"]


def _nn_one_unit(args):
    unit_id, pcs_flat, labels, metric_params = args

    try:
        nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(pcs_flat, labels, unit_id, **metric_params)
    except:
        nn_hit_rate = np.nan
        nn_miss_rate = np.nan

    return unit_id, nn_hit_rate, nn_miss_rate


def _nearest_neighbor_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    nn_result = namedtuple("NearestNeighborResult", ["nn_hit_rate", "nn_miss_rate"])

    # Use pre-computed PCA data
    pca_data_per_unit = tmp_data["pca_data_per_unit"]

    # Extract job parameters
    n_jobs = job_kwargs.get("n_jobs", 1)
    progress_bar = job_kwargs.get("progress_bar", False)
    mp_context = job_kwargs.get("mp_context", None)

    nn_hit_rate_dict = {}
    nn_miss_rate_dict = {}

    if n_jobs == 1:
        # Sequential processing
        units_loop = unit_ids
        if progress_bar:
            from tqdm.auto import tqdm

            units_loop = tqdm(units_loop, desc="Nearest neighbor metrics")

        for unit_id in units_loop:
            pcs_flat = pca_data_per_unit[unit_id]["pcs_flat"]
            labels = pca_data_per_unit[unit_id]["labels"]

            try:
                nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(pcs_flat, labels, unit_id, **metric_params)
            except:
                nn_hit_rate = np.nan
                nn_miss_rate = np.nan

            nn_hit_rate_dict[unit_id] = nn_hit_rate
            nn_miss_rate_dict[unit_id] = nn_miss_rate
    else:
        # Parallel processing
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        import warnings
        import platform

        print(f"computing nearest neighbor metrics with n_jobs={n_jobs}, mp_context={mp_context}")

        if mp_context is not None and platform.system() == "Windows":
            assert mp_context != "fork", "'fork' mp_context not supported on Windows!"
        elif mp_context == "fork" and platform.system() == "Darwin":
            warnings.warn('As of Python 3.8 "fork" is no longer considered safe on macOS')

        # Prepare arguments - only pass pickle-able data
        args_list = []
        for unit_id in unit_ids:
            pcs_flat = pca_data_per_unit[unit_id]["pcs_flat"]
            labels = pca_data_per_unit[unit_id]["labels"]
            args_list.append((unit_id, pcs_flat, labels, metric_params))

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context(mp_context) if mp_context else None,
        ) as executor:
            results = executor.map(_nn_one_unit, args_list)
            if progress_bar:
                from tqdm.auto import tqdm

                results = tqdm(results, total=len(unit_ids), desc="Nearest neighbor metrics")

            for unit_id, nn_hit_rate, nn_miss_rate in results:
                nn_hit_rate_dict[unit_id] = nn_hit_rate
                nn_miss_rate_dict[unit_id] = nn_miss_rate

    return nn_result(nn_hit_rate=nn_hit_rate_dict, nn_miss_rate=nn_miss_rate_dict)


class NearestNeighborMetrics(BaseMetric):
    metric_name = "nearest_neighbor"
    metric_function = _nearest_neighbor_metric_function
    metric_params = {"max_spikes": 10000, "n_neighbors": 5}
    metric_columns = ["nn_hit_rate", "nn_miss_rate"]
    metric_dtypes = {"nn_hit_rate": float, "nn_miss_rate": float}
    depend_on = ["principal_components"]


def _nn_advanced_one_unit(args):
    unit_id, sorting_analyzer, n_spikes_all_units, fr_all_units, metric_params, seed = args

    nn_isolation_params = {
        k: v
        for k, v in metric_params.items()
        if k
        in [
            "max_spikes",
            "min_spikes",
            "min_fr",
            "n_neighbors",
            "n_components",
            "radius_um",
            "peak_sign",
            "min_spatial_overlap",
        ]
    }
    nn_noise_params = {
        k: v
        for k, v in metric_params.items()
        if k in ["max_spikes", "min_spikes", "min_fr", "n_neighbors", "n_components", "radius_um", "peak_sign"]
    }

    # NN Isolation
    try:
        nn_isolation, nn_unit_id = nearest_neighbors_isolation(
            sorting_analyzer,
            unit_id,
            n_spikes_all_units=n_spikes_all_units,
            fr_all_units=fr_all_units,
            seed=seed,
            **nn_isolation_params,
        )
    except:
        nn_isolation, nn_unit_id = np.nan, np.nan

    # NN Noise Overlap
    try:
        nn_noise_overlap = nearest_neighbors_noise_overlap(
            sorting_analyzer,
            unit_id,
            n_spikes_all_units=n_spikes_all_units,
            fr_all_units=fr_all_units,
            seed=seed,
            **nn_noise_params,
        )
    except:
        nn_noise_overlap = np.nan

    return unit_id, nn_isolation, nn_unit_id, nn_noise_overlap


def _nn_advanced_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    nn_advanced_result = namedtuple("NNAdvancedResult", ["nn_isolation", "nn_unit_id", "nn_noise_overlap"])

    # Use pre-computed data
    n_spikes_all_units = tmp_data["n_spikes_all_units"]
    fr_all_units = tmp_data["fr_all_units"]

    # Extract job parameters
    n_jobs = job_kwargs.get("n_jobs", 1)
    progress_bar = job_kwargs.get("progress_bar", False)
    mp_context = job_kwargs.get("mp_context", None)
    seed = job_kwargs.get("seed", None)

    nn_isolation_dict = {}
    nn_unit_id_dict = {}
    nn_noise_overlap_dict = {}

    if n_jobs == 1:
        # Sequential processing
        units_loop = unit_ids
        if progress_bar:
            from tqdm.auto import tqdm

            units_loop = tqdm(units_loop, desc="Advanced NN metrics")

        for unit_id in units_loop:
            _, nn_isolation, nn_unit_id, nn_noise_overlap = _nn_advanced_one_unit(
                (unit_id, sorting_analyzer, n_spikes_all_units, fr_all_units, metric_params, seed)
            )
            nn_isolation_dict[unit_id] = nn_isolation
            nn_unit_id_dict[unit_id] = nn_unit_id
            nn_noise_overlap_dict[unit_id] = nn_noise_overlap
    else:
        # Parallel processing
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        import warnings
        import platform

        if mp_context is not None and platform.system() == "Windows":
            assert mp_context != "fork", "'fork' mp_context not supported on Windows!"
        elif mp_context == "fork" and platform.system() == "Darwin":
            warnings.warn('As of Python 3.8 "fork" is no longer considered safe on macOS')

        # Prepare arguments
        args_list = []
        for unit_id in unit_ids:
            args_list.append((unit_id, sorting_analyzer, n_spikes_all_units, fr_all_units, metric_params, seed))

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context(mp_context) if mp_context else None,
        ) as executor:
            results = executor.map(_nn_advanced_one_unit, args_list)
            if progress_bar:
                from tqdm.auto import tqdm

                results = tqdm(results, total=len(unit_ids), desc="Advanced NN metrics")

            for unit_id, nn_isolation, nn_unit_id, nn_noise_overlap in results:
                nn_isolation_dict[unit_id] = nn_isolation
                nn_unit_id_dict[unit_id] = nn_unit_id
                nn_noise_overlap_dict[unit_id] = nn_noise_overlap

    return nn_advanced_result(
        nn_isolation=nn_isolation_dict, nn_unit_id=nn_unit_id_dict, nn_noise_overlap=nn_noise_overlap_dict
    )


class NearestNeighborAdvancedMetrics(BaseMetric):
    metric_name = "nn_advanced"
    metric_function = _nn_advanced_metric_function
    metric_params = {
        "max_spikes": 1000,
        "min_spikes": 10,
        "min_fr": 0.0,
        "n_neighbors": 4,
        "n_components": 10,
        "radius_um": 100,
        "peak_sign": "neg",
        "min_spatial_overlap": 0.5,
    }
    metric_columns = ["nn_isolation", "nn_unit_id", "nn_noise_overlap"]
    metric_dtypes = {"nn_isolation": float, "nn_unit_id": "object", "nn_noise_overlap": float}
    depend_on = ["principal_components", "waveforms", "templates"]


def _silhouette_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    silhouette_result = namedtuple("SilhouetteResult", ["silhouette"])

    # Use pre-computed PCA data
    pca_data_per_unit = tmp_data["pca_data_per_unit"]

    silhouette_dict = {}
    method = metric_params.get("method", "simplified")

    for unit_id in unit_ids:
        pcs_flat = pca_data_per_unit[unit_id]["pcs_flat"]
        labels = pca_data_per_unit[unit_id]["labels"]

        try:
            if method == "simplified":
                silhouette_value = simplified_silhouette_score(pcs_flat, labels, unit_id)
            else:  # method == "full"
                silhouette_value = silhouette_score(pcs_flat, labels, unit_id)
        except:
            silhouette_value = np.nan

        silhouette_dict[unit_id] = silhouette_value

    return silhouette_result(silhouette=silhouette_dict)


class SilhouetteMetrics(BaseMetric):
    metric_name = "silhouette"
    metric_function = _silhouette_metric_function
    metric_params = {"method": "simplified"}
    metric_columns = ["silhouette"]
    metric_dtypes = {"silhouette": float}
    depend_on = ["principal_components"]


pca_metrics = [
    MahalanobisMetrics,
    DPrimeMetrics,
    NearestNeighborMetrics,
    SilhouetteMetrics,
    NearestNeighborAdvancedMetrics,
]
