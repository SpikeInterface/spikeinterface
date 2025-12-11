from __future__ import annotations

# """Sorting components: clustering"""
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

class PositionsClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    _default_params = {
        "peak_locations": None,
        "peak_localization_kwargs": {"method": "center_of_mass"},
        "hdbscan_kwargs": {"min_cluster_size": 20, "allow_single_cluster": True, "core_dist_n_jobs": -1},
    }

    name = "hdbscan_positions"
    need_noise_levels = False
    params_doc = """
        peak_locations : The locations of the peaks
        peaks_localization_kwargs : dict
            Kwargs for peak localization if locations are not provided
        hdbscan_kwargs: dict
            Kwargs for HDBSCAN
    """

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        if params["peak_locations"] is None:
            from spikeinterface.sortingcomponents.peak_localization import localize_peaks

            peak_locations = localize_peaks(
                recording, peaks, **params["peak_localization_kwargs"], job_kwargs=job_kwargs
            )
        else:
            peak_locations = params["peak_locations"]

        location_keys = ["x", "y"]
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)

        clustering = HDBSCAN(**params["hdbscan_kwargs"]).fit(locations)
        peak_labels = clustering.labels_

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        return labels, peak_labels, dict()
