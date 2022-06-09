# """Sorting components: clustering"""
from pathlib import Path
import numpy as np

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

from spikeinterface.sortingcomponents.peak_waveform_features import (
    compute_waveform_features_peaks,
)


class PositionPTPScaledClustering:
    """
    hdbscan clustering on peak_locations and scaled logptps
    """

    _default_params = {
        "peak_locations": None,
        "ptps": None,
        "scales": (1, 1, 30),
        "peak_localization_kwargs": {"method": "center_of_mass"},
        "hdbscan_kwargs": {
            "min_cluster_size": 20,
            "min_samples": 20,
            "allow_single_cluster": True,
        },
        "debug": False,
        "tmp_folder": None,
    }

    @classmethod
    def main_function(cls, recording, peaks, params, **job_kwargs):
        assert HAVE_HDBSCAN, "position clustering need hdbscan to be installed"
        d = params

        if d["peak_locations"] is None:
            from spikeinterface.sortingcomponents.peak_localization import (
                localize_peaks,
            )

            peak_locations = localize_peaks(
                recording, peaks, **d["peak_localization_kwargs"]
            )
        else:
            peak_locations = d["peak_locations"]

        tmp_folder = d["tmp_folder"]
        if tmp_folder is not None:
            tmp_folder.mkdir(exist_ok=True)

        location_keys = ["x", "y"]
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)

        if d["ptps"] is None:
            ptps = compute_waveform_features_peaks(
                recording,
                peaks,
                time_range_list=[(1, 1.5)],
                feature_list=["ptps"],
                **job_kwargs,
            )[:, :, 0]
        else:
            ptps = d["ptps"]

        maxptps = np.max(ptps, 1)
        logmaxptps = np.log(maxptps)

        to_cluster_from = np.hstack((locations, logmaxptps[:, np.newaxis]))
        to_cluster_from = to_cluster_from * d["scales"]

        clustering = hdbscan.hdbscan(to_cluster_from, **d["hdbscan_kwargs"])
        peak_labels = clustering[0]

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        if d["debug"]:
            import matplotlib.pyplot as plt
            import spikeinterface.full as si

            fig1, ax = plt.subplots()
            kwargs = dict(
                probe_shape_kwargs=dict(
                    facecolor="w", edgecolor="k", lw=0.5, alpha=0.3
                ),
                contacts_kargs=dict(alpha=0.5, edgecolor="k", lw=0.5, facecolor="w"),
            )
            si.plot_probe_map(recording, ax=ax, **kwargs)
            ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, color="k")

            fig2, ax = plt.subplots()
            si.plot_probe_map(recording, ax=ax, **kwargs)
            ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, c=peak_labels)

            if tmp_folder is not None:
                fig1.savefig(tmp_folder / "peak_locations.png")
                fig2.savefig(tmp_folder / "peak_locations_clustered.png")

        return labels, peak_labels
