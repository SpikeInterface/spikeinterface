from __future__ import annotations

import numpy as np
from collections import namedtuple

from spikeinterface.core.analyzer_extension_core import BaseMetric


def get_trough_and_peak_idx(template):
    """
    Return the indices into the input template of the detected trough
    (minimum of template) and peak (maximum of template, after trough).
    Assumes negative trough and positive peak.

    Parameters
    ----------
    template: numpy.ndarray
        The 1D template waveform

    Returns
    -------
    trough_idx: int
        The index of the trough
    peak_idx: int
        The index of the peak
    """
    assert template.ndim == 1
    trough_idx = np.argmin(template)
    peak_idx = trough_idx + np.argmax(template[trough_idx:])
    return trough_idx, peak_idx


#########################################################################################
# Single-channel metrics
def get_peak_to_valley(template_single, sampling_frequency, trough_idx=None, peak_idx=None, **kwargs) -> float:
    """
    Return the peak to valley duration in seconds of input waveforms.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak

    Returns
    -------
    ptv: float
        The peak to valley duration in seconds
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)
    ptv = (peak_idx - trough_idx) / sampling_frequency
    return ptv


def get_peak_trough_ratio(template_single, sampling_frequency=None, trough_idx=None, peak_idx=None, **kwargs) -> float:
    """
    Return the peak to trough ratio of input waveforms.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak

    Returns
    -------
    ptratio: float
        The peak to trough ratio
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)
    ptratio = template_single[peak_idx] / template_single[trough_idx]
    return ptratio


def get_half_width(template_single, sampling_frequency, trough_idx=None, peak_idx=None, **kwargs) -> float:
    """
    Return the half width of input waveforms in seconds.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough
    peak_idx: int, default: None
        The index of the peak

    Returns
    -------
    hw: float
        The half width in seconds
    """
    if trough_idx is None or peak_idx is None:
        trough_idx, peak_idx = get_trough_and_peak_idx(template_single)

    if peak_idx == 0:
        return np.nan

    trough_val = template_single[trough_idx]
    # threshold is half of peak height (assuming baseline is 0)
    threshold = 0.5 * trough_val

    (cpre_idx,) = np.where(template_single[:trough_idx] < threshold)
    (cpost_idx,) = np.where(template_single[trough_idx:] < threshold)

    if len(cpre_idx) == 0 or len(cpost_idx) == 0:
        hw = np.nan

    else:
        # last occurence of template lower than thr, before peak
        cross_pre_pk = cpre_idx[0] - 1
        # first occurence of template lower than peak, after peak
        cross_post_pk = cpost_idx[-1] + 1 + trough_idx

        hw = (cross_post_pk - cross_pre_pk) / sampling_frequency
    return hw


def get_repolarization_slope(template_single, sampling_frequency, trough_idx=None, **kwargs):
    """
    Return slope of repolarization period between trough and baseline

    After reaching it's maximum polarization, the neuron potential will
    recover. The repolarization slope is defined as the dV/dT of the action potential
    between trough and baseline. The returned slope is in units of (unit of template)
    per second. By default traces are scaled to units of uV, controlled
    by `sorting_analyzer.return_in_uV`. In this case this function returns the slope
    in uV/s.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    trough_idx: int, default: None
        The index of the trough

    Returns
    -------
    slope: float
        The repolarization slope
    """
    if trough_idx is None:
        trough_idx, _ = get_trough_and_peak_idx(template_single)

    times = np.arange(template_single.shape[0]) / sampling_frequency

    if trough_idx == 0:
        return np.nan

    (rtrn_idx,) = np.nonzero(template_single[trough_idx:] >= 0)
    if len(rtrn_idx) == 0:
        return np.nan
    # first time after trough, where template is at baseline
    return_to_base_idx = rtrn_idx[0] + trough_idx

    if return_to_base_idx - trough_idx < 3:
        return np.nan

    import scipy.stats

    res = scipy.stats.linregress(times[trough_idx:return_to_base_idx], template_single[trough_idx:return_to_base_idx])
    return res.slope


def get_recovery_slope(template_single, sampling_frequency, peak_idx=None, **kwargs):
    """
    Return the recovery slope of input waveforms. After repolarization,
    the neuron hyperpolarizes until it peaks. The recovery slope is the
    slope of the action potential after the peak, returning to the baseline
    in dV/dT. The returned slope is in units of (unit of template)
    per second. By default traces are scaled to units of uV, controlled
    by `sorting_analyzer.return_in_uV`. In this case this function returns the slope
    in uV/s. The slope is computed within a user-defined window after the peak.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    peak_idx: int, default: None
        The index of the peak
    **kwargs: Required kwargs:
        - recovery_window_ms: the window in ms after the peak to compute the recovery_slope

    Returns
    -------
    res.slope: float
        The recovery slope
    """
    import scipy.stats

    assert "recovery_window_ms" in kwargs, "recovery_window_ms must be given as kwarg"
    recovery_window_ms = kwargs["recovery_window_ms"]
    if peak_idx is None:
        _, peak_idx = get_trough_and_peak_idx(template_single)

    times = np.arange(template_single.shape[0]) / sampling_frequency

    if peak_idx == 0:
        return np.nan
    max_idx = int(peak_idx + ((recovery_window_ms / 1000) * sampling_frequency))
    max_idx = np.min([max_idx, template_single.shape[0]])

    res = scipy.stats.linregress(times[peak_idx:max_idx], template_single[peak_idx:max_idx])
    return res.slope


def get_number_of_peaks(template_single, sampling_frequency, **kwargs):
    """
    Count the total number of peaks (positive + negative) in the template.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - peak_relative_threshold: the relative threshold to detect positive and negative peaks
        - peak_width_ms: the width in samples to detect peaks

    Returns
    -------
    number_of_peaks: int
        the total number of peaks (positive + negative)
    """
    from scipy.signal import find_peaks

    assert "peak_relative_threshold" in kwargs, "peak_relative_threshold must be given as kwarg"
    assert "peak_width_ms" in kwargs, "peak_width_ms must be given as kwarg"
    peak_relative_threshold = kwargs["peak_relative_threshold"]
    peak_width_ms = kwargs["peak_width_ms"]
    max_value = np.max(np.abs(template_single))
    peak_width_samples = int(peak_width_ms / 1000 * sampling_frequency)

    pos_peaks = find_peaks(template_single, height=peak_relative_threshold * max_value, width=peak_width_samples)
    neg_peaks = find_peaks(-template_single, height=peak_relative_threshold * max_value, width=peak_width_samples)
    num_positive = len(pos_peaks[0])
    num_negative = len(neg_peaks[0])
    return num_positive, num_negative


#########################################################################################
# Multi-channel metrics
def transform_column_range(template, channel_locations, column_range, depth_direction="y"):
    """
    Transform template and channel locations based on column range.
    """
    column_dim = 0 if depth_direction == "y" else 1
    if column_range is None:
        template_column_range = template
        channel_locations_column_range = channel_locations
    else:
        max_channel_x = channel_locations[np.argmax(np.ptp(template, axis=0)), 0]
        column_mask = np.abs(channel_locations[:, column_dim] - max_channel_x) <= column_range
        template_column_range = template[:, column_mask]
        channel_locations_column_range = channel_locations[column_mask]
    return template_column_range, channel_locations_column_range


def sort_template_and_locations(template, channel_locations, depth_direction="y"):
    """
    Sort template and locations.
    """
    depth_dim = 1 if depth_direction == "y" else 0
    sort_indices = np.argsort(channel_locations[:, depth_dim])
    return template[:, sort_indices], channel_locations[sort_indices, :]


def fit_velocity(peak_times, channel_dist):
    """
    Fit velocity from peak times and channel distances using robust Theilsen estimator.
    """
    # from scipy.stats import linregress
    # slope, intercept, _, _, _ = linregress(peak_times, channel_dist)

    from sklearn.linear_model import TheilSenRegressor

    theil = TheilSenRegressor()
    theil.fit(peak_times.reshape(-1, 1), channel_dist)
    slope = theil.coef_[0]
    intercept = theil.intercept_
    score = theil.score(peak_times.reshape(-1, 1), channel_dist)
    return slope, intercept, score


def get_velocity_fits(template, channel_locations, sampling_frequency, **kwargs):
    """
    Compute both velocity above and below the max channel of the template in units um/ms.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - min_channels: the minimum number of channels above or below to compute velocity
        - min_r2: the minimum r2 to accept the velocity fit
        - column_range: the range in um in the x-direction to consider channels for velocity

    Returns
    -------
    velocity_above : float
        The velocity above the max channel
    velocity_below : float
        The velocity below the max channel
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    assert "min_channels" in kwargs, "min_channels must be given as kwarg"
    assert "min_r2" in kwargs, "min_r2 must be given as kwarg"
    assert "column_range" in kwargs, "column_range must be given as kwarg"

    depth_direction = kwargs["depth_direction"]
    min_channels_for_velocity = kwargs["min_channels"]
    min_r2 = kwargs["min_r2"]
    column_range = kwargs["column_range"]

    depth_dim = 1 if depth_direction == "y" else 0
    template, channel_locations = transform_column_range(template, channel_locations, column_range, depth_direction)
    template, channel_locations = sort_template_and_locations(template, channel_locations, depth_direction)

    # find location of max channel
    max_sample_idx, max_channel_idx = np.unravel_index(np.argmin(template), template.shape)
    max_peak_time = max_sample_idx / sampling_frequency * 1000
    max_channel_location = channel_locations[max_channel_idx]

    # Compute velocity above
    channels_above = channel_locations[:, depth_dim] >= max_channel_location[depth_dim]
    if np.sum(channels_above) < min_channels_for_velocity:
        velocity_above = np.nan
    else:
        template_above = template[:, channels_above]
        channel_locations_above = channel_locations[channels_above]
        peak_times_ms_above = np.argmin(template_above, 0) / sampling_frequency * 1000 - max_peak_time
        distances_um_above = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations_above])
        velocity_above, _, score = fit_velocity(peak_times_ms_above, distances_um_above)
        if score < min_r2:
            velocity_above = np.nan

    # Compute velocity below
    channels_below = channel_locations[:, depth_dim] <= max_channel_location[depth_dim]
    if np.sum(channels_below) < min_channels_for_velocity:
        velocity_below = np.nan
    else:
        template_below = template[:, channels_below]
        channel_locations_below = channel_locations[channels_below]
        peak_times_ms_below = np.argmin(template_below, 0) / sampling_frequency * 1000 - max_peak_time
        distances_um_below = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations_below])
        velocity_below, _, score = fit_velocity(peak_times_ms_below, distances_um_below)
        if score < min_r2:
            velocity_below = np.nan

    return velocity_above, velocity_below


def get_exp_decay(template, channel_locations, sampling_frequency=None, **kwargs):
    """
    Compute the exponential decay of the template amplitude over distance in units um/s.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - peak_function: the function to use to compute the peak amplitude for the exp decay ("ptp" or "min")
        - min_r2: the minimum r2 to accept the exp decay fit

    Returns
    -------
    exp_decay_value : float
        The exponential decay of the template amplitude
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    def exp_decay(x, decay, amp0, offset):
        return amp0 * np.exp(-decay * x) + offset

    assert "peak_function" in kwargs, "peak_function must be given as kwarg"
    peak_function = kwargs["peak_function"]
    assert "min_r2" in kwargs, "min_r2 must be given as kwarg"
    min_r2 = kwargs["min_r2"]
    # exp decay fit
    if peak_function == "ptp":
        fun = np.ptp
    elif peak_function == "min":
        fun = np.min
    peak_amplitudes = np.abs(fun(template, axis=0))
    max_channel_location = channel_locations[np.argmax(peak_amplitudes)]
    channel_distances = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations])
    distances_sort_indices = np.argsort(channel_distances)

    # longdouble is float128 when the platform supports it, otherwise it is float64
    channel_distances_sorted = channel_distances[distances_sort_indices].astype(np.longdouble)
    peak_amplitudes_sorted = peak_amplitudes[distances_sort_indices].astype(np.longdouble)

    try:
        amp0 = peak_amplitudes_sorted[0]
        offset0 = np.min(peak_amplitudes_sorted)

        popt, _ = curve_fit(
            exp_decay,
            channel_distances_sorted,
            peak_amplitudes_sorted,
            bounds=([1e-5, amp0 - 0.5 * amp0, 0], [2, amp0 + 0.5 * amp0, 2 * offset0]),
            p0=[1e-3, peak_amplitudes_sorted[0], offset0],
        )
        r2 = r2_score(peak_amplitudes_sorted, exp_decay(channel_distances_sorted, *popt))
        exp_decay_value = popt[0]

        if r2 < min_r2:
            exp_decay_value = np.nan
    except:
        exp_decay_value = np.nan

    return exp_decay_value


def get_spread(template, channel_locations, sampling_frequency, **kwargs) -> float:
    """
    Compute the spread of the template amplitude over distance in units um/s.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - spread_threshold: the threshold to compute the spread
        - column_range: the range in um in the x-direction to consider channels for velocity

    Returns
    -------
    spread : float
        Spread of the template amplitude
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    depth_direction = kwargs["depth_direction"]
    assert "spread_threshold" in kwargs, "spread_threshold must be given as kwarg"
    spread_threshold = kwargs["spread_threshold"]
    assert "spread_smooth_um" in kwargs, "spread_smooth_um must be given as kwarg"
    spread_smooth_um = kwargs["spread_smooth_um"]
    assert "column_range" in kwargs, "column_range must be given as kwarg"
    column_range = kwargs["column_range"]

    depth_dim = 1 if depth_direction == "y" else 0
    template, channel_locations = transform_column_range(template, channel_locations, column_range)
    template, channel_locations = sort_template_and_locations(template, channel_locations, depth_direction)

    MM = np.ptp(template, 0)
    channel_depths = channel_locations[:, depth_dim]

    if spread_smooth_um is not None and spread_smooth_um > 0:
        from scipy.ndimage import gaussian_filter1d

        spread_sigma = spread_smooth_um / np.median(np.diff(np.unique(channel_depths)))
        MM = gaussian_filter1d(MM, spread_sigma)

    MM = MM / np.max(MM)

    channel_locations_above_threshold = channel_locations[MM > spread_threshold]
    channel_depth_above_threshold = channel_locations_above_threshold[:, depth_dim]

    spread = np.ptp(channel_depth_above_threshold)

    return spread


def single_channel_metric(unit_function, sorting_analyzer, unit_ids, tmp_data, **metric_params):
    result = {}
    templates_single = tmp_data["templates_single"]
    troughs = tmp_data.get("troughs", None)
    peaks = tmp_data.get("peaks", None)
    sampling_frequency = tmp_data["sampling_frequency"]
    for unit_index, unit_id in enumerate(unit_ids):
        template_single = templates_single[unit_index]
        trough_idx = troughs[unit_id] if troughs is not None else None
        peak_idx = peaks[unit_id] if peaks is not None else None
        metric_params["trough_idx"] = trough_idx
        metric_params["peak_idx"] = peak_idx
        value = unit_function(template_single, sampling_frequency, **metric_params)
        result[unit_id] = value
    return result


class PeakToValley(BaseMetric):
    metric_name = "peak_to_valley"
    metric_params = {}
    metric_columns = {"peak_to_valley": float}
    metric_descriptions = {
        "peak_to_valley": "Duration in s between the trough (minimum) and the peak (maximum) of the spike waveform."
    }
    needs_tmp_data = True

    @staticmethod
    def _peak_to_valley_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return single_channel_metric(
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
    metric_columns = {"peak_trough_ratio": float}
    metric_descriptions = {
        "peak_trough_ratio": "Ratio of the amplitude of the peak (maximum) to the trough (minimum) of the spike waveform."
    }
    needs_tmp_data = True

    @staticmethod
    def _peak_to_trough_ratio_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return single_channel_metric(
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
    metric_descriptions = {
        "half_width": "Duration in s at half the amplitude of the trough (minimum) of the spike waveform."
    }
    needs_tmp_data = True

    @staticmethod
    def _half_width_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return single_channel_metric(
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
    metric_descriptions = {
        "repolarization_slope": "Slope of the repolarization phase of the spike waveform, between the trough (minimum) and return to baseline in uV/s."
    }
    needs_tmp_data = True

    @staticmethod
    def _repolarization_slope_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return single_channel_metric(
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
    metric_descriptions = {
        "recovery_slope": "Slope of the recovery phase of the spike waveform, after the peak (maximum) returning to baseline in uV/s."
    }
    needs_tmp_data = True

    @staticmethod
    def _recovery_slope_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return single_channel_metric(
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
    for unit_index, unit_id in enumerate(unit_ids):
        template_single = templates_single[unit_index]
        num_positive, num_negative = get_number_of_peaks(template_single, sampling_frequency, **metric_params)
        num_positive_peaks_dict[unit_id] = num_positive
        num_negative_peaks_dict[unit_id] = num_negative
    return num_peaks_result(num_positive_peaks=num_positive_peaks_dict, num_negative_peaks=num_negative_peaks_dict)


class NumberOfPeaks(BaseMetric):
    metric_name = "number_of_peaks"
    metric_function = _number_of_peaks_metric_function
    metric_params = {"peak_relative_threshold": 0.2, "peak_width_ms": 0.1}
    metric_columns = {"num_positive_peaks": int, "num_negative_peaks": int}
    metric_descriptions = {
        "num_positive_peaks": "Number of positive peaks in the template",
        "num_negative_peaks": "Number of negative peaks in the template",
    }
    needs_tmp_data = True


single_channel_metrics = [
    PeakToValley,
    PeakToTroughRatio,
    HalfWidth,
    RepolarizationSlope,
    RecoverySlope,
    NumberOfPeaks,
]


def _get_velocity_fits_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
    velocity_above_result = namedtuple("Velocities", ["velocity_above", "velocity_below"])
    velocity_above_dict = {}
    velocity_below_dict = {}
    templates_multi = tmp_data["templates_multi"]
    channel_locations_multi = tmp_data["channel_locations_multi"]
    sampling_frequency = tmp_data["sampling_frequency"]
    metric_params["depth_direction"] = tmp_data["depth_direction"]
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
        "min_channels": 3,
        "min_r2": 0.2,
        "column_range": None,
    }
    metric_columns = {"velocity_above": float, "velocity_below": float}
    metric_descriptions = {
        "velocity_above": "Velocity of the spike propagation above the max channel in um/ms",
        "velocity_below": "Velocity of the spike propagation below the max channel in um/ms",
    }
    needs_tmp_data = True


def multi_channel_metric(unit_function, sorting_analyzer, unit_ids, tmp_data, **metric_params):
    result = {}
    templates_multi = tmp_data["templates_multi"]
    channel_locations_multi = tmp_data["channel_locations_multi"]
    sampling_frequency = tmp_data["sampling_frequency"]
    metric_params["depth_direction"] = tmp_data["depth_direction"]
    for unit_index, unit_id in enumerate(unit_ids):
        channel_locations = channel_locations_multi[unit_index]
        template = templates_multi[unit_index]
        value = unit_function(template, channel_locations, sampling_frequency, **metric_params)
        result[unit_id] = value
    return result


class ExpDecay(BaseMetric):
    metric_name = "exp_decay"
    metric_params = {"peak_function": "ptp", "min_r2": 0.2}
    metric_columns = {"exp_decay": float}
    metric_descriptions = {
        "exp_decay": ("Exponential decay of the template amplitude over distance from the extremum channel (1/um).")
    }
    needs_tmp_data = True

    @staticmethod
    def _exp_decay_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return multi_channel_metric(
            unit_function=get_exp_decay,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _exp_decay_metric_function


class Spread(BaseMetric):
    metric_name = "spread"
    metric_params = {"spread_threshold": 0.5, "spread_smooth_um": 20, "column_range": None}
    metric_columns = {"spread": float}
    metric_descriptions = {
        "spread": (
            "Spread of the template amplitude in um, calculated as the distance between channels whose "
            "templates exceed the spread_threshold."
        )
    }
    needs_tmp_data = True

    @staticmethod
    def _spread_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return multi_channel_metric(
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
