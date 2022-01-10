from .misc_metrics import (
    compute_num_spikes,
    compute_firing_rate,
    compute_presence_ratio,
    compute_snrs,
    compute_isi_violations,
    compute_amplitudes_cutoff,
)

from .pca_metrics import (
    calculate_pc_metrics,
    mahalanobis_metrics,
    lda_metrics,
    nearest_neighbors_metrics,
    nearest_neighbors_isolation,
    nearest_neighbors_noise_overlap)

from .pca_metrics import _possible_pc_metric_names

# list of all available metrics and mapping to function
# this list MUST NOT contain pca metrics
# the list is handle separatly
_metric_name_to_func = {
    "num_spikes": compute_num_spikes,
    "firing_rate": compute_firing_rate,
    "presence_ratio": compute_presence_ratio,
    "snr": compute_snrs,
    "isi_violation": compute_isi_violations,
    "amplitude_cutoff": compute_amplitudes_cutoff,
    # 'mahalanobis_metrics': mahalanobis_metrics,
    # 'd_prime': lda_metrics,
    # 'nearest_neighbor': nearest_neighbors_metrics,
    # 'nn_isolation': nearest_neighbors_isolation,
    # 'nn_noise_overlap': nearest_neighbors_noise_overlap
}

# TODO
# @Cole @ Alessio
# "silhouette_score",
# "max_drift",
# "cumulative_drift",
