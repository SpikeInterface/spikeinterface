from .misc_metrics import (
    compute_num_spikes,
    compute_firing_rate,
    compute_presence_ratio,
    compute_snrs,
    compute_isi_violations,
    compute_amplitudes_cutoff,
)

from .pca_metrics import calculate_pc_metrics, _possible_pc_metric_names

# based on PCA
# "isolation_distance", "l_ratio", "d_prime",  "nn_hit_rate", "nn_miss_rate",


# in misc_metrics.py
_metric_name_to_func = {
    "num_spikes": compute_num_spikes,
    "firing_rate": compute_firing_rate,
    "presence_ratio": compute_presence_ratio,
    "snr": compute_snrs,
    "isi_violation": compute_isi_violations,
    "amplitude_cutoff": compute_amplitudes_cutoff,
}

# TODO
# @Cole @ Alessio
# "silhouette_score",
# "noise_overlap",
# "max_drift",
# "cumulative_drift",
