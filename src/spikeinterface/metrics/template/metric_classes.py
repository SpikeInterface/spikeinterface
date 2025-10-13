from __future__ import annotations

from collections import namedtuple
from spikeinterface.core.analyzer_extension_core import BaseMetric
from spikeinterface.metrics.template.metrics_implementations import (
    get_peak_to_valley,
    get_peak_trough_ratio,
    get_half_width,
    get_repolarization_slope,
    get_recovery_slope,
    get_number_of_peaks,
    get_exp_decay,
    get_spread,
    get_velocity_fits,
    get_trough_and_peak_idx,
)


def single_channel_metric(unit_function, sorting_analyzer, unit_ids, tmp_data, **metric_params):
    result = {}
    templates_single = tmp_data["templates_single"]
    troughs = tmp_data.get("troughs", None)
    peaks = tmp_data.get("peaks", None)
    sampling_frequency = tmp_data["sampling_frequency"]
    for unit_id in unit_ids:
        template_single = templates_single[sorting_analyzer.sorting.id_to_index(unit_id)]
        trough_idx = troughs[unit_id] if troughs is not None else None
        peak_idx = peaks[unit_id] if peaks is not None else None
        value = unit_function(template_single, sampling_frequency, trough_idx, peak_idx, **metric_params)
        result[unit_id] = value
    return result


class PeakToValley(BaseMetric):
    metric_name = "peak_to_valley"
    metric_params = {}
    metric_columns = {"peak_to_valley": float}
    needs_tmp_data = True

    @staticmethod
    def _peak_to_valley_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        single_channel_metric(
            unit_function=get_peak_to_valley,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _peak_to_valley_metric_function


class PeakToTroughRatio(BaseMetric):
    metric_name = "peak_trough_ratio"
    metric_params = {}
    metric_columns = {"peak_to_trough_ratio": float}
    needs_tmp_data = True

    @staticmethod
    def _peak_to_trough_ratio_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        single_channel_metric(
            unit_function=get_peak_trough_ratio,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _peak_to_trough_ratio_metric_function


class HalfWidth(BaseMetric):
    metric_name = "half_width"
    metric_params = {}
    metric_columns = {"half_width": float}
    needs_tmp_data = True

    @staticmethod
    def _half_width_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        single_channel_metric(
            unit_function=get_half_width,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _half_width_metric_function


class RepolarizationSlope(BaseMetric):
    metric_name = "repolarization_slope"
    metric_params = {}
    metric_columns = {"repolarization_slope": float}
    needs_tmp_data = True

    @staticmethod
    def _repolarization_slope_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        single_channel_metric(
            unit_function=get_repolarization_slope,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _repolarization_slope_metric_function


class RecoverySlope(BaseMetric):
    metric_name = "recovery_slope"
    metric_params = {"recovery_window_ms": 0.7}
    metric_columns = {"recovery_slope": float}
    needs_tmp_data = True

    @staticmethod
    def _recovery_slope_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        single_channel_metric(
            unit_function=get_recovery_slope,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _recovery_slope_metric_function


def _number_of_peaks_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
    num_peaks_result = namedtuple("NumberOfPeaksResult", ["num_positive_peaks", "num_negative_peaks"])
    num_positive_peaks_dict = {}
    num_negative_peaks_dict = {}
    sampling_frequency = sorting_analyzer.sampling_frequency
    templates_single = tmp_data["templates_single"]
    for unit_id in unit_ids:
        template_single = templates_single[sorting_analyzer.sorting.id_to_index(unit_id)]
        num_positive, num_negative = get_number_of_peaks(template_single, sampling_frequency, **metric_params)
        num_positive_peaks_dict[unit_id] = num_positive
        num_negative_peaks_dict[unit_id] = num_negative
    return num_peaks_result(num_positive_peaks=num_positive_peaks_dict, num_negative_peaks=num_negative_peaks_dict)


class NumberOfPeaks(BaseMetric):
    metric_name = "number_of_peaks"
    metric_function = _number_of_peaks_metric_function
    metric_params = {"peak_relative_threshold": 0.2, "peak_width_ms": 0.1}
    metric_columns = {"num_positive_peaks": int, "num_negative_peaks": int}
    needs_tmp_data = True


single_channel_metrics = [
    PeakToValley,
    PeakToTroughRatio,
    HalfWidth,
    RepolarizationSlope,
    RecoverySlope,
    NumberOfPeaks,
]


def _get_velocity_fits_metric_function(sorting_analyzer, unit_ids, tmp_data, metric_params, job_kwargs):
    velocity_above_result = namedtuple("Velocities", ["velocity_above", "velocity_below"])
    velocity_above_dict = {}
    velocity_below_dict = {}
    templates_multi = tmp_data["templates_multi"]
    channel_locations_multi = tmp_data["channel_locations_multi"]
    sampling_frequency = tmp_data["sampling_frequency"]
    for unit_index, unit_id in enumerate(unit_ids):
        channel_locations = channel_locations_multi[unit_index]
        template = templates_multi[unit_index]
        vel_above, vel_below = get_velocity_fits(template, channel_locations, sampling_frequency, **metric_params)
        velocity_above_dict[unit_id] = vel_above
        velocity_below_dict[unit_id] = vel_below
    return velocity_above_result(velocity_above=velocity_above_dict, velocity_below=velocity_below_dict)


class VelocityFits(BaseMetric):
    metric_name = "velocity_fits"
    metric_function = _get_velocity_fits_metric_function
    metric_params = {
        "depth_direction": "y",
        "min_channels_for_velocity": 3,
        "min_r2_velocity": 0.2,
        "column_range": None,
    }
    metric_columns = {"velocity_above": float, "velocity_below": float}
    needs_tmp_data = True


def multi_channel_metric(unit_function, sorting_analyzer, unit_ids, tmp_data, **metric_params):
    result = {}
    templates_multi = tmp_data["templates_multi"]
    channel_locations_multi = tmp_data["channel_locations_multi"]
    sampling_frequency = tmp_data["sampling_frequency"]
    for unit_index, unit_id in enumerate(unit_ids):
        channel_locations = channel_locations_multi[unit_index]
        template = templates_multi[unit_index]
        value = unit_function(template, channel_locations, sampling_frequency, **metric_params)
        result[unit_id] = value
    return result


class ExpDecay(BaseMetric):
    metric_name = "exp_decay"
    metric_params = {"exp_peak_function": "ptp", "min_r2_exp_decay": 0.2}
    metric_columns = {"exp_decay": float}

    @staticmethod
    def _exp_decay_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        multi_channel_metric(
            unit_function=get_exp_decay,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _exp_decay_metric_function


class Spread(BaseMetric):
    metric_name = "spread"
    metric_params = {"depth_direction": "y", "spread_threshold": 0.5, "spread_smooth_um": 20, "column_range": None}
    metric_columns = {"spread": float}

    @staticmethod
    def _spread_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        multi_channel_metric(
            unit_function=get_spread,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _spread_metric_function


multi_channel_metrics = [
    VelocityFits,
    ExpDecay,
    Spread,
]
