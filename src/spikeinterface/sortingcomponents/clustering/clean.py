from __future__ import annotations

import numpy as np

from .tools import FeaturesLoader, compute_template_from_sparse

# This is work in progress ...


def clean_clusters(
    peaks,
    peak_labels,
    recording,
    features_dict_or_folder,
    peak_sign="neg",
):
    total_channels = recording.get_num_channels()

    if isinstance(features_dict_or_folder, dict):
        features = features_dict_or_folder
    else:
        features = FeaturesLoader(features_dict_or_folder)

    clean_labels = peak_labels.copy()

    sparse_wfs = features["sparse_wfs"]
    sparse_mask = features["sparse_mask"]

    labels_set = np.setdiff1d(peak_labels, [-1]).tolist()
    n = len(labels_set)

    count = np.zeros(n, dtype="int64")
    for i, label in enumerate(labels_set):
        count[i] = np.sum(peak_labels == label)

    templates = compute_template_from_sparse(peaks, peak_labels, labels_set, sparse_wfs, sparse_mask, total_channels)

    if peak_sign == "both":
        max_values = np.max(np.abs(templates), axis=(1, 2))
    elif peak_sign == "neg":
        max_values = -np.min(templates, axis=(1, 2))
    elif peak_sign == "pos":
        max_values = np.max(templates, axis=(1, 2))

    return clean_labels
