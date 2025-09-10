from __future__ import annotations

import numpy as np

from spikeinterface.qualitymetrics.quality_metric_list import metric_extension_dependencies


def _has_required_extensions(sorting_analyzer, metric_name):

    required_extensions = metric_extension_dependencies[metric_name]

    not_computed_required_extensions = []
    for ext in required_extensions:
        if all(sorting_analyzer.has_extension(name) is False for name in ext.split("|")):
            not_computed_required_extensions.append(ext)

    if len(not_computed_required_extensions) > 0:
        warnings_string = f"The `{metric_name}` metric requires the {not_computed_required_extensions} extensions.\n"
        warnings_string += "Use the sorting_analyzer.compute(["
        for count, ext in enumerate(not_computed_required_extensions):
            if count == len(not_computed_required_extensions) - 1:
                warnings_string += f"'{ext}'"
            else:
                warnings_string += f"'{ext}', "
        warnings_string += f"]) method to compute."
        raise ValueError(warnings_string)


def create_ground_truth_pc_distributions(center_locations, total_points):
    """
    Simulate PCs as multivariate Gaussians, for testing PC-based quality metrics
    Values are created for only one channel and vary along one dimension.

    Parameters
    ----------
    center_locations : array-like (units, ) or (channels, units)
        Mean of the multivariate gaussian at each channel for each unit.
    total_points : array-like
        Number of points in each unit distribution.

    Returns
    -------
    all_pcs : numpy.ndarray
        PC scores for each point.
    all_labels : numpy.array
        Labels for each point.
    """
    from scipy.stats import multivariate_normal

    np.random.seed(0)

    if len(np.array(center_locations).shape) == 1:
        distributions = [
            multivariate_normal.rvs(mean=[center, 0.0, 0.0], cov=[1.0, 1.0, 1.0], size=size)
            for center, size in zip(center_locations, total_points)
        ]
        all_pcs = np.concatenate(distributions, axis=0)

    else:
        all_pcs = np.empty((np.sum(total_points), 3, center_locations.shape[0]))
        for channel in range(center_locations.shape[0]):
            distributions = [
                multivariate_normal.rvs(mean=[center, 0.0, 0.0], cov=[1.0, 1.0, 1.0], size=size)
                for center, size in zip(center_locations[channel], total_points)
            ]
            all_pcs[:, :, channel] = np.concatenate(distributions, axis=0)

    all_labels = np.concatenate([np.ones((total_points[i],), dtype="int") * i for i in range(len(total_points))])

    return all_pcs, all_labels
