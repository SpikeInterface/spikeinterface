"""Lists of quality metrics."""

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
    compute_drift_metrics
)

from .pca_metrics import (
    calculate_pc_metrics,
    mahalanobis_metrics,
    lda_metrics,
    nearest_neighbors_metrics,
    nearest_neighbors_isolation,
    nearest_neighbors_noise_overlap
)

from .pca_metrics import _possible_pc_metric_names


# list of all available metrics and mapping to function
# this list MUST NOT contain pca metrics, which are handled separately
_misc_metric_name_to_func = {
    "num_spikes" : compute_num_spikes,
    "firing_rate" : compute_firing_rates,
    "presence_ratio" : compute_presence_ratios,
    "snr" : compute_snrs,
    "isi_violation" : compute_isi_violations,
    "rp_violation" : compute_refrac_period_violations,
    "sliding_rp_violation" : compute_sliding_rp_violations,
    "amplitude_cutoff" : compute_amplitude_cutoffs,
    "amplitude_median" : compute_amplitude_medians,
    "drift" : compute_drift_metrics
}

