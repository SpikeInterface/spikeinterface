"""Lists of quality metrics."""

from __future__ import annotations


from .misc_metrics import (
    compute_num_spikes,
    compute_firing_rates,
    compute_presence_ratios,
    compute_snrs,
    compute_isi_violations,
    compute_refrac_period_violations,
    compute_sliding_rp_violations,
    compute_amplitude_cutoffs,
    compute_amplitude_medians,
    compute_drift_metrics,
    compute_synchrony_metrics,
    compute_firing_ranges,
    compute_amplitude_cv_metrics,
    compute_sd_ratio,
)

from .pca_metrics import (
    compute_pc_metrics,
    calculate_pc_metrics,  # remove after 0.103.0
    mahalanobis_metrics,
    lda_metrics,
    nearest_neighbors_metrics,
    nearest_neighbors_isolation,
    nearest_neighbors_noise_overlap,
    silhouette_score,
    simplified_silhouette_score,
)

from .pca_metrics import _possible_pc_metric_names


# list of all available metrics and mapping to function
# this list MUST NOT contain pca metrics, which are handled separately
_misc_metric_name_to_func = {
    "num_spikes": compute_num_spikes,
    "firing_rate": compute_firing_rates,
    "presence_ratio": compute_presence_ratios,
    "snr": compute_snrs,
    "isi_violation": compute_isi_violations,
    "rp_violation": compute_refrac_period_violations,
    "sliding_rp_violation": compute_sliding_rp_violations,
    "amplitude_cutoff": compute_amplitude_cutoffs,
    "amplitude_median": compute_amplitude_medians,
    "amplitude_cv": compute_amplitude_cv_metrics,
    "synchrony": compute_synchrony_metrics,
    "firing_range": compute_firing_ranges,
    "drift": compute_drift_metrics,
    "sd_ratio": compute_sd_ratio,
}

# a dict converting the name of the metric for computation to the output of that computation
qm_compute_name_to_column_names = {
    "num_spikes": ["num_spikes"],
    "firing_rate": ["firing_rate"],
    "presence_ratio": ["presence_ratio"],
    "snr": ["snr"],
    "isi_violation": ["isi_violations_ratio", "isi_violations_count"],
    "rp_violation": ["rp_violations", "rp_contamination"],
    "sliding_rp_violation": ["sliding_rp_violation"],
    "amplitude_cutoff": ["amplitude_cutoff"],
    "amplitude_median": ["amplitude_median"],
    "amplitude_cv": ["amplitude_cv_median", "amplitude_cv_range"],
    "synchrony": [
        "sync_spike_2",
        "sync_spike_4",
        "sync_spike_8",
    ],
    "firing_range": ["firing_range"],
    "drift": ["drift_ptp", "drift_std", "drift_mad"],
    "sd_ratio": ["sd_ratio"],
    "isolation_distance": ["isolation_distance"],
    "l_ratio": ["l_ratio"],
    "d_prime": ["d_prime"],
    "nearest_neighbor": ["nn_hit_rate", "nn_miss_rate"],
    "nn_isolation": ["nn_isolation", "nn_unit_id"],
    "nn_noise_overlap": ["nn_noise_overlap"],
    "silhouette": ["silhouette"],
    "silhouette_full": ["silhouette_full"],
}

# this dict allows us to ensure the appropriate dtype of metrics rather than allow Pandas to infer them
column_name_to_column_dtype = {
    "num_spikes": int,
    "firing_rate": float,
    "presence_ratio": float,
    "snr": float,
    "isi_violations_ratio": float,
    "isi_violations_count": float,
    "rp_violations": float,
    "rp_contamination": float,
    "sliding_rp_violation": float,
    "amplitude_cutoff": float,
    "amplitude_median": float,
    "amplitude_cv_median": float,
    "amplitude_cv_range": float,
    "sync_spike_2": float,
    "sync_spike_4": float,
    "sync_spike_8": float,
    "firing_range": float,
    "drift_ptp": float,
    "drift_std": float,
    "drift_mad": float,
    "sd_ratio": float,
    "isolation_distance": float,
    "l_ratio": float,
    "d_prime": float,
    "nn_hit_rate": float,
    "nn_miss_rate": float,
    "nn_isolation": float,
    "nn_unit_id": float,
    "nn_noise_overlap": float,
    "silhouette": float,
    "silhouette_full": float,
}
