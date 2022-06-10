import numpy as np
from scipy.spatial import KDTree


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
    tree = KDTree(feats)
    dist, _ = tree.query(feats, k=6)
    dist = dist[:, 1:]
    if ptp_weighting:
        dist = np.sum(
            c * np.log(dist) + np.log(1 / (scales[2] * np.log(maxptps)))[:, None], 1
        )
    idx_keep = dist <= np.percentile(dist, threshold)
    return idx_keep
