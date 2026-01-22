from __future__ import annotations

import numpy as np
from spikeinterface.core.base import unit_period_dtype


def compute_bin_edges_per_unit(sorting, segment_samples, bin_duration_s=1.0, periods=None, concatenated=True):
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
    concatenated : bool, default: True
        Wheter the bins are concatenated across segments or not.
        If False, the bin edges are computed per segment and the first index of each segment is 0.
        If True, the bin edges are computed on the concatenated segments, with the correct offsets.

    Returns
    -------
    dict
        Bin edges for each unit. If concatenated is True, the bin edges are a 1D array.
        If False, the bin edges are a list of arrays, one per segment.
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
                    if not concatenated:
                        bin_edges.append(np.array([]))
                    continue
                seg_start = np.sum(segment_samples[:seg_index]) if concatenated else 0
                bin_edges_segment = []
                for period in seg_periods:
                    start_sample = seg_start + period["start_sample_index"]
                    end_sample = seg_start + period["end_sample_index"]
                    end_sample = end_sample // bin_duration_samples * bin_duration_samples + 1  # align to bin
                    bin_edges_segment.extend(np.arange(start_sample, end_sample, bin_duration_samples))
                bin_edges_segment = np.unique(np.array(bin_edges_segment))
                if concatenated:
                    bin_edges.extend(bin_edges_segment)
                else:
                    bin_edges.append(bin_edges_segment)
            bin_edges_for_units[unit_id] = bin_edges
    else:
        for unit_id in sorting.unit_ids:
            bin_edges = []
            for seg_index in range(num_segments):
                seg_start = np.sum(segment_samples[:seg_index]) if concatenated else 0
                seg_end = seg_start + segment_samples[seg_index]
                # for segments which are not the last, we don't need to correct the end
                # since the first index of the next segment will be the end of the current segment
                if seg_index == num_segments - 1:
                    seg_end = seg_end // bin_duration_samples * bin_duration_samples + 1  # align to bin
                bin_edges_segment = np.arange(seg_start, seg_end, bin_duration_samples)
                if concatenated:
                    bin_edges.extend(bin_edges_segment)
                else:
                    bin_edges.append(bin_edges_segment)
            bin_edges_for_units[unit_id] = bin_edges
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
        total_samples_array = np.zeros(len(sorting_analyzer.unit_ids), dtype="int64")
        sorting = sorting_analyzer.sorting
        for period in periods:
            unit_index = period["unit_index"]
            total_samples_array[unit_index] += period["end_sample_index"] - period["start_sample_index"]
        total_samples = dict(zip(sorting.unit_ids, total_samples_array))
    else:
        total = sorting_analyzer.get_total_samples()
        total_samples = {unit_id: total for unit_id in sorting_analyzer.unit_ids}
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


def create_regular_periods(sorting_analyzer, num_periods, bin_size_s=None):
    """
    Computes and sets periods for each unit in the sorting analyzer.
    The periods span the total duration of the recording, but divide it into
    smaller periods either by specifying the number of periods or the size of each bin.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The sorting analyzer containing the units and recording information.
    num_periods : int
        The number of periods to divide the total duration into (used if bin_size_s is None).
    bin_size_s : float, defaut: None
        If given, periods will be multiple of this size in seconds.

    Returns
    -------
    periods
        np.ndarray of dtype unit_period_dtype containing the segment, start, end samples and unit index.
    """
    all_periods = []
    for segment_index in range(sorting_analyzer.recording.get_num_segments()):
        samples_per_period = sorting_analyzer.get_num_samples(segment_index) // num_periods
        if bin_size_s is not None:
            bin_size_samples = int(bin_size_s * sorting_analyzer.sampling_frequency)
            samples_per_period = samples_per_period // bin_size_samples * bin_size_samples
            num_periods = int(np.round(sorting_analyzer.get_num_samples(segment_index) / samples_per_period))
        for unit_index, unit_id in enumerate(sorting_analyzer.unit_ids):
            period_starts = np.arange(0, sorting_analyzer.get_num_samples(segment_index), samples_per_period)
            periods_per_unit = np.zeros(len(period_starts), dtype=unit_period_dtype)
            for i, period_start in enumerate(period_starts):
                period_end = min(period_start + samples_per_period, sorting_analyzer.get_num_samples(segment_index))
                periods_per_unit[i]["segment_index"] = segment_index
                periods_per_unit[i]["start_sample_index"] = period_start
                periods_per_unit[i]["end_sample_index"] = period_end
                periods_per_unit[i]["unit_index"] = unit_index
            all_periods.append(periods_per_unit)
    return np.concatenate(all_periods)


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
