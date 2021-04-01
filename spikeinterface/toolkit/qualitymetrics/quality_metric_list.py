from .misc_metrics import (
    compute_num_spikes,
    compute_firing_rate,
    compute_presence_ratio,
    compute_snrs,
    compute_isi_violations,
    compute_amplitudes_cuoff,
    )

_metric_name_to_func = {
    # misc
    "num_spikes": compute_num_spikes,
    "firing_rate": compute_firing_rate,
    "presence_ratio": compute_presence_ratio,
    "snr": compute_snrs,
    "isi_violation": compute_isi_violations,
    "amplitude_cutoff": compute_amplitudes_cuoff,

    
    #~ "max_drift",
    #~ "cumulative_drift",
    #~ "silhouette_score",
    #~ "isolation_distance",
    #~ "l_ratio",
     #~ "d_prime", 
     #~ "noise_overlap",
     #~ "nn_hit_rate",
     #~ "nn_miss_rate",
}