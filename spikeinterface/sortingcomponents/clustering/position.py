# """Sorting components: clustering"""
from pathlib import Path

import numpy as np
try:
    import hdbscan
    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False

class PositionClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """
    _default_params = {
        "peak_locations" : None,
        "hdbscan_params": {"min_cluster_size" : 20,  "allow_single_cluster" : True, 'metric' : 'l2'},
        "probability_thr" : 0,
        "apply_norm" : True,
        "debug" : False,
        "tmp_folder" : None,
    }

    @classmethod
    def main_function(cls, recording, peaks, params):
        d = params

        assert d['peak_locations'] is not None, "peak_locations should not be None!"

        tmp_folder = d['tmp_folder']
        if tmp_folder is not None:
            tmp_folder.mkdir(exist_ok=True)

        peak_locations = d['peak_locations']
        
        location_keys = ['x', 'y']
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)
        
        if d['apply_norm']:
            locations_n = locations.copy()
            locations_n -= np.mean(locations_n, axis=0)
            locations_n /= np.std(locations_n, axis=0)
        else:
            locations_n = locations
        
        clustering = hdbscan.hdbscan(locations_n, **d['hdbscan_params'])
        peak_labels = clustering[0]
        cluster_probability = clustering[2]
        
        # keep only persistent
        persistent_clusters, = np.nonzero(clustering[2] > d['probability_thr'])
        mask = np.in1d(peak_labels, persistent_clusters)
        peak_labels[~mask] = -2
        
        labels = np.unique(peak_labels)
        labels = labels[labels>=0]

        if d['debug']:
            import matplotlib.pyplot as plt
            import spikeinterface.full as si
            fig1, ax = plt.subplots()
            kwargs = dict(probe_shape_kwargs=dict(facecolor='w', edgecolor='k', lw=0.5, alpha=0.3),
                                    contacts_kargs = dict(alpha=0.5, edgecolor='k', lw=0.5, facecolor='w'))
            si.plot_probe_map(recording, ax=ax, **kwargs)
            ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, color='k')

            fig2, ax = plt.subplots()
            si.plot_probe_map(recording, ax=ax, **kwargs)
            ax.scatter(locations[:, 0], locations[:, 1], alpha=0.5, s=1, c=peak_labels)

            if tmp_folder is not None:
                fig1.savefig(tmp_folder / 'peak_locations.png')
                fig2.savefig(tmp_folder / 'peak_locations_clustered.png')

        return labels, peak_labels
