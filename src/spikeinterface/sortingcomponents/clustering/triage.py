from __future__ import annotations

import numpy as np


def nearest_neighor_triage(
    x,
    y,
    maxptps,
    scales=(1, 1, 30),
    threshold=80,
    c=1,
    ptp_weighting=True,
):
    feats = np.c_[scales[0] * x, scales[1] * y, scales[2] * np.log(maxptps)]
    from scipy.spatial import KDTree

    tree = KDTree(feats)
    dist, _ = tree.query(feats, k=6)
    dist = dist[:, 1:]
    log_dist = c * np.log(dist)
    if ptp_weighting:
        log_dist += np.log(1 / (scales[2] * np.log(maxptps)))[:, None]
    dist = np.sum(log_dist, 1)
    idx_keep = dist <= np.percentile(dist, threshold)
    return idx_keep
