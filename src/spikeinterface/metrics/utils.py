from __future__ import annotations

import numpy as np


def compute_bin_edges_per_unit(sorting, segment_samples, bin_duration_s=1.0, periods=None):
    """
    Compute bin edges for units, optionally taking into account periods.

    Parameters
    ----------
    sorting : Sorting
        Sorting object containing unit information.
    segment_samples : list or array-like
        Number of samples in each segment.
    bin_duration_s : float, default: 1
        Duration of each bin in seconds
    periods : array of unit_period_dtype, default: None
        Periods to consider for each unit
    """
    bin_edges_for_units = {}
    num_segments = len(segment_samples)
    bin_duration_samples = int(bin_duration_s * sorting.sampling_frequency)

    if periods is not None:
        for unit_id in sorting.unit_ids:
            unit_index = sorting.id_to_index(unit_id)
            periods_unit = periods[periods["unit_index"] == unit_index]
            bin_edges = []
            for seg_index in range(num_segments):
                seg_periods = periods_unit[periods_unit["segment_index"] == seg_index]
                if len(seg_periods) == 0:
                    continue
                seg_start = np.sum(segment_samples[:seg_index])
                for period in seg_periods:
                    start_sample = seg_start + period["start_sample_index"]
                    end_sample = seg_start + period["end_sample_index"]
                    end_sample = end_sample // bin_duration_samples * bin_duration_samples + 1  # align to bin
                    bin_edges.extend(np.arange(start_sample, end_sample, bin_duration_samples))
            bin_edges_for_units[unit_id] = np.unique(np.array(bin_edges))
    else:
        for unit_id in sorting.unit_ids:
            bin_edges = []
            for seg_index in range(num_segments):
                seg_start = np.sum(segment_samples[:seg_index])
                seg_end = seg_start + segment_samples[seg_index]
                # for segments which are not the last, we don't need to correct the end
                # since the first index of the next segment will be the end of the current segment
                if seg_index == num_segments - 1:
                    seg_end = seg_end // bin_duration_samples * bin_duration_samples + 1  # align to bin
                bins = np.arange(seg_start, seg_end, bin_duration_samples)
                bin_edges.extend(bins)
            bin_edges_for_units[unit_id] = np.array(bin_edges)
    return bin_edges_for_units


def compute_total_samples_per_unit(sorting_analyzer, periods=None):
    """
    Get total number of samples for each unit, optionally taking into account periods.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object.
    periods : array of unit_period_dtype, default: None
        Periods to consider for each unit.

    Returns
    -------
    dict
        Total number of samples for each unit.
    """
    if periods is not None:
        total_samples = {}
        sorting = sorting_analyzer.sorting
        for unit_id in sorting.unit_ids:
            unit_index = sorting.id_to_index(unit_id)
            periods_unit = periods[periods["unit_index"] == unit_index]
            num_samples_in_period = 0
            for period in periods_unit:
                num_samples_in_period += period["end_sample_index"] - period["start_sample_index"]
            total_samples[unit_id] = num_samples_in_period
    else:
        total_samples = {unit_id: sorting_analyzer.get_total_samples() for unit_id in sorting_analyzer.unit_ids}
    return total_samples


def compute_total_durations_per_unit(sorting_analyzer, periods=None):
    """
    Compute total duration for each unit, optionally taking into account periods.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer object.
    periods : array of unit_period_dtype, default: None
        Periods to consider for each unit.

    Returns
    -------
    dict
        Total duration for each unit.
    """
    total_samples = compute_total_samples_per_unit(sorting_analyzer, periods=periods)
    total_durations = {
        unit_id: samples / sorting_analyzer.sampling_frequency for unit_id, samples in total_samples.items()
    }
    return total_durations


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
