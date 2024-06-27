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
