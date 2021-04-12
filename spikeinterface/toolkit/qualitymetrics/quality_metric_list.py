from .misc_metrics import (
    compute_num_spikes,
    compute_firing_rate,
    compute_presence_ratio,
    compute_snrs,
    compute_isi_violations,
    compute_amplitudes_cutoff,
    )

_metric_name_to_func = {
    # misc
    "num_spikes": compute_num_spikes,
    "firing_rate": compute_firing_rate,
    "presence_ratio": compute_presence_ratio,
    "snr": compute_snrs,
    "isi_violation": compute_isi_violations,
    "amplitude_cutoff": compute_amplitudes_cutoff,
    
    # based on PCA
    #~ "isolation_distance",
    #~ "l_ratio",
    #~ "d_prime", 
    #~ "nn_hit_rate",
    #~ "nn_miss_rate",
    #~ "silhouette_score",
    #~ "noise_overlap",

    # TODO
    #~ "max_drift",
    #~ "cumulative_drift",
}