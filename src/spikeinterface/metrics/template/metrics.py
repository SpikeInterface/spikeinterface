from __future__ import annotations

import numpy as np
from spikeinterface.core.analyzer_extension_core import BaseMetric


def get_trough_and_peak_idx(
    template, min_thresh_detect_peaks_troughs=0.4, smooth=True, smooth_window_frac=0.1, smooth_polyorder=3
):
    """
    Detect troughs and peaks in a template waveform and return detailed information
    about each detected feature.

    Parameters
    ----------
    template : numpy.ndarray
        The 1D template waveform
    min_thresh_detect_peaks_troughs : float, default: 0.4
        Minimum prominence threshold as a fraction of the template's absolute max value
    smooth : bool, default: True
        Whether to apply smoothing before peak detection
    smooth_window_frac : float, default: 0.1
        Smoothing window length as a fraction of template length (0.05-0.2 recommended)
    smooth_polyorder : int, default: 3
        Polynomial order for Savitzky-Golay filter (must be < window_length)

    Returns
    -------
    troughs : dict
        Dictionary containing:
        - "indices": array of all trough indices
        - "values": array of all trough values
        - "prominences": array of all trough prominences
        - "widths": array of all trough widths
        - "main_idx": index of the main trough (most prominent)
        - "main_loc": location (sample index) of the main trough in template
    peaks_before : dict
        Dictionary containing peaks detected before the main trough (initial peaks):
        - "indices": array of all peak indices (in original template coordinates)
        - "values": array of all peak values
        - "prominences": array of all peak prominences
        - "widths": array of all peak widths
        - "main_idx": index of the main peak (most prominent)
        - "main_loc": location (sample index) of the main peak in template
    peaks_after : dict
        Dictionary containing peaks detected after the main trough (repolarization peaks):
        - "indices": array of all peak indices (in original template coordinates)
        - "values": array of all peak values
        - "prominences": array of all peak prominences
        - "widths": array of all peak widths
        - "main_idx": index of the main peak (most prominent)
        - "main_loc": location (sample index) of the main peak in template
    """
    from scipy.signal import find_peaks, savgol_filter

    assert template.ndim == 1

    # Smooth template to reduce noise while preserving peaks using Savitzky-Golay filter
    if smooth:
        window_length = int(len(template) * smooth_window_frac) // 2 * 2 + 1
        window_length = max(smooth_polyorder + 2, window_length)  # Must be > polyorder
        template = savgol_filter(template, window_length=window_length, polyorder=smooth_polyorder)

    # Initialize empty result dictionaries
    empty_dict = {
        "indices": np.array([], dtype=int),
        "values": np.array([]),
        "prominences": np.array([]),
        "widths": np.array([]),
        "main_idx": None,
        "main_loc": None,
    }

    # Get min prominence to detect peaks and troughs relative to template abs max value
    min_prominence = min_thresh_detect_peaks_troughs * np.nanmax(np.abs(template))

    # --- Find troughs (by inverting waveform and using find_peaks) ---
    trough_locs, trough_props = find_peaks(-template, prominence=min_prominence, width=0)

    if len(trough_locs) == 0:
        # Fallback: use global minimum
        trough_locs = np.array([np.nanargmin(template)])
        trough_props = {"prominences": np.array([np.nan]), "widths": np.array([np.nan])}

    # Determine main trough (most prominent, or first if no valid prominences)
    trough_prominences = trough_props.get("prominences", np.array([]))
    if len(trough_prominences) > 0 and not np.all(np.isnan(trough_prominences)):
        main_trough_idx = np.nanargmax(trough_prominences)
    else:
        main_trough_idx = 0

    main_trough_loc = trough_locs[main_trough_idx]

    troughs = {
        "indices": trough_locs,
        "values": template[trough_locs],
        "prominences": trough_props.get("prominences", np.full(len(trough_locs), np.nan)),
        "widths": trough_props.get("widths", np.full(len(trough_locs), np.nan)),
        "main_idx": main_trough_idx,
        "main_loc": main_trough_loc,
    }

    # --- Find peaks before the main trough ---
    if main_trough_loc > 3:
        template_before = template[:main_trough_loc]

        # Try with original prominence
        peak_locs_before, peak_props_before = find_peaks(template_before, prominence=min_prominence, width=0)

        # If no peaks found, try with lower prominence (keep only max peak)
        if len(peak_locs_before) == 0:
            lower_prominence = 0.075 * min_thresh_detect_peaks_troughs * np.nanmax(np.abs(template))
            peak_locs_before, peak_props_before = find_peaks(template_before, prominence=lower_prominence, width=0)
            # Keep only the most prominent peak when using lower threshold
            if len(peak_locs_before) > 1:
                prominences = peak_props_before.get("prominences", np.array([]))
                if len(prominences) > 0 and not np.all(np.isnan(prominences)):
                    max_idx = np.nanargmax(prominences)
                    peak_locs_before = np.array([peak_locs_before[max_idx]])
                    peak_props_before = {
                        "prominences": np.array([prominences[max_idx]]),
                        "widths": np.array([peak_props_before.get("widths", np.array([np.nan]))[max_idx]]),
                    }

        # If still no peaks found, fall back to argmax
        if len(peak_locs_before) == 0:
            peak_locs_before = np.array([np.nanargmax(template_before)])
            peak_props_before = {"prominences": np.array([np.nan]), "widths": np.array([np.nan])}

        peak_prominences_before = peak_props_before.get("prominences", np.array([]))
        if len(peak_prominences_before) > 0 and not np.all(np.isnan(peak_prominences_before)):
            main_peak_before_idx = np.nanargmax(peak_prominences_before)
        else:
            main_peak_before_idx = 0

        peaks_before = {
            "indices": peak_locs_before,
            "values": template[peak_locs_before],
            "prominences": peak_props_before.get("prominences", np.full(len(peak_locs_before), np.nan)),
            "widths": peak_props_before.get("widths", np.full(len(peak_locs_before), np.nan)),
            "main_idx": main_peak_before_idx,
            "main_loc": peak_locs_before[main_peak_before_idx],
        }
    else:
        peaks_before = empty_dict.copy()

    # --- Find peaks after the main trough (repolarization peaks) ---
    if main_trough_loc < len(template) - 3:
        template_after = template[main_trough_loc:]

        # Try with original prominence
        peak_locs_after, peak_props_after = find_peaks(template_after, prominence=min_prominence, width=0)

        # If no peaks found, try with lower prominence (keep only max peak)
        if len(peak_locs_after) == 0:
            lower_prominence = 0.075 * min_thresh_detect_peaks_troughs * np.nanmax(np.abs(template))
            peak_locs_after, peak_props_after = find_peaks(template_after, prominence=lower_prominence, width=0)
            # Keep only the most prominent peak when using lower threshold
            if len(peak_locs_after) > 1:
                prominences = peak_props_after.get("prominences", np.array([]))
                if len(prominences) > 0 and not np.all(np.isnan(prominences)):
                    max_idx = np.nanargmax(prominences)
                    peak_locs_after = np.array([peak_locs_after[max_idx]])
                    peak_props_after = {
                        "prominences": np.array([prominences[max_idx]]),
                        "widths": np.array([peak_props_after.get("widths", np.array([np.nan]))[max_idx]]),
                    }

        # If still no peaks found, fall back to argmax
        if len(peak_locs_after) == 0:
            peak_locs_after = np.array([np.nanargmax(template_after)])
            peak_props_after = {"prominences": np.array([np.nan]), "widths": np.array([np.nan])}

        # Convert to original template coordinates
        peak_locs_after_abs = peak_locs_after + main_trough_loc

        peak_prominences_after = peak_props_after.get("prominences", np.array([]))
        if len(peak_prominences_after) > 0 and not np.all(np.isnan(peak_prominences_after)):
            main_peak_after_idx = np.nanargmax(peak_prominences_after)
        else:
            main_peak_after_idx = 0

        peaks_after = {
            "indices": peak_locs_after_abs,
            "values": template[peak_locs_after_abs],
            "prominences": peak_props_after.get("prominences", np.full(len(peak_locs_after), np.nan)),
            "widths": peak_props_after.get("widths", np.full(len(peak_locs_after), np.nan)),
            "main_idx": main_peak_after_idx,
            "main_loc": peak_locs_after_abs[main_peak_after_idx],
        }
    else:
        peaks_after = empty_dict.copy()

    return troughs, peaks_before, peaks_after


def get_main_to_next_peak_duration(template, sampling_frequency, troughs, peaks_before, peaks_after, **kwargs):
    """
    Calculate duration from the main extremum to the next extremum.

    The duration is measured from the largest absolute feature (main trough or main peak)
    to the next extremum. For typical negative-first waveforms, this is trough-to-peak.
    For positive-first waveforms, this is peak-to-trough.

    Parameters
    ----------
    template : numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency in Hz
    troughs : dict
        Trough detection results from get_trough_and_peak_idx
    peaks_before : dict
        Peak before trough results from get_trough_and_peak_idx
    peaks_after : dict
        Peak after trough results from get_trough_and_peak_idx

    Returns
    -------
    main_to_next_peak_duration_us : float
        Duration in microseconds from main extremum to next extremum
    """

    # Get main locations and values
    trough_loc = troughs["main_loc"]
    trough_val = template[trough_loc] if trough_loc is not None else None

    peak_before_loc = peaks_before["main_loc"]
    peak_before_val = template[peak_before_loc] if peak_before_loc is not None else None

    peak_after_loc = peaks_after["main_loc"]
    peak_after_val = template[peak_after_loc] if peak_after_loc is not None else None

    # Find the main extremum (largest absolute value)
    candidates = []
    if trough_loc is not None and trough_val is not None:
        candidates.append(("trough", trough_loc, abs(trough_val)))
    if peak_before_loc is not None and peak_before_val is not None:
        candidates.append(("peak_before", peak_before_loc, abs(peak_before_val)))
    if peak_after_loc is not None and peak_after_val is not None:
        candidates.append(("peak_after", peak_after_loc, abs(peak_after_val)))

    if len(candidates) == 0:
        return np.nan

    # Sort by absolute value to find main extremum
    candidates.sort(key=lambda x: x[2], reverse=True)
    main_type, main_loc, _ = candidates[0]

    # Find the next extremum after the main one
    if main_type == "trough":
        # Main is trough, next is peak_after
        if peak_after_loc is not None:
            duration_samples = abs(peak_after_loc - main_loc)
        elif peak_before_loc is not None:
            duration_samples = abs(main_loc - peak_before_loc)
        else:
            return np.nan
    elif main_type == "peak_before":
        # Main is peak before, next is trough
        if trough_loc is not None:
            duration_samples = abs(trough_loc - main_loc)
        else:
            return np.nan
    else:  # peak_after
        # Main is peak after, previous is trough
        if trough_loc is not None:
            duration_samples = abs(main_loc - trough_loc)
        else:
            return np.nan

    # Convert to microseconds
    main_to_next_peak_duration_us = (duration_samples / sampling_frequency) * 1e6

    return main_to_next_peak_duration_us


def get_waveform_ratios(template, troughs, peaks_before, peaks_after, **kwargs):
    """
    Calculate various waveform amplitude ratios.

    Parameters
    ----------
    template : numpy.ndarray
        The 1D template waveform
    troughs : dict
        Trough detection results from get_trough_and_peak_idx
    peaks_before : dict
        Peak before trough results from get_trough_and_peak_idx
    peaks_after : dict
        Peak after trough results from get_trough_and_peak_idx

    Returns
    -------
    ratios : dict
        Dictionary containing:
        - "peak_before_to_trough_ratio": ratio of peak before to trough amplitude
        - "peak_after_to_trough_ratio": ratio of peak after to trough amplitude
        - "peak_before_to_peak_after_ratio": ratio of peak before to peak after amplitude
        - "main_peak_to_trough_ratio": ratio of larger peak to trough amplitude
    """
    # Get absolute amplitudes
    trough_amp = abs(template[troughs["main_loc"]]) if troughs["main_loc"] is not None else np.nan
    peak_before_amp = abs(template[peaks_before["main_loc"]]) if peaks_before["main_loc"] is not None else np.nan
    peak_after_amp = abs(template[peaks_after["main_loc"]]) if peaks_after["main_loc"] is not None else np.nan

    def safe_ratio(a, b):
        if np.isnan(a) or np.isnan(b) or b == 0:
            return np.nan
        return a / b

    ratios = {
        "peak_before_to_trough_ratio": safe_ratio(peak_before_amp, trough_amp),
        "peak_after_to_trough_ratio": safe_ratio(peak_after_amp, trough_amp),
        "peak_before_to_peak_after_ratio": safe_ratio(peak_before_amp, peak_after_amp),
        "main_peak_to_trough_ratio": safe_ratio(
            (
                max(peak_before_amp, peak_after_amp)
                if not (np.isnan(peak_before_amp) and np.isnan(peak_after_amp))
                else np.nan
            ),
            trough_amp,
        ),
    }

    return ratios


def get_waveform_baseline_flatness(template, sampling_frequency, **kwargs):
    """
    Compute the baseline flatness of the waveform.

    This metric measures the ratio of the max absolute amplitude in the baseline
    window to the max absolute amplitude of the whole waveform. A lower value
    indicates a flat baseline (expected for good units).

    Parameters
    ----------
    template : numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency in Hz
    **kwargs : Required kwargs:
        - baseline_window_ms : tuple of (start_ms, end_ms) defining the baseline window
          relative to waveform start. Default is (0, 0.5) for first 0.5ms.

    Returns
    -------
    baseline_flatness : float
        Ratio of max(abs(baseline)) / max(abs(waveform)). Lower = flatter baseline.
    """
    baseline_window_ms = kwargs.get("baseline_window_ms", (0.0, 0.5))

    if baseline_window_ms is None:
        return np.nan

    start_ms, end_ms = baseline_window_ms
    start_idx = int(start_ms / 1000 * sampling_frequency)
    end_idx = int(end_ms / 1000 * sampling_frequency)

    # Clamp to valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(template), end_idx)

    if end_idx <= start_idx:
        return np.nan

    baseline_segment = template[start_idx:end_idx]

    if len(baseline_segment) == 0:
        return np.nan

    max_baseline = np.nanmax(np.abs(baseline_segment))
    max_waveform = np.nanmax(np.abs(template))

    if max_waveform == 0 or np.isnan(max_waveform):
        return np.nan

    baseline_flatness = max_baseline / max_waveform

    return baseline_flatness


def get_waveform_widths(template, sampling_frequency, troughs, peaks_before, peaks_after, **kwargs):
    """
    Get the widths of the main trough and peaks in microseconds.

    Parameters
    ----------
    template : numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency in Hz
    troughs : dict
        Trough detection results from get_trough_and_peak_idx
    peaks_before : dict
        Peak before trough results from get_trough_and_peak_idx
    peaks_after : dict
        Peak after trough results from get_trough_and_peak_idx

    Returns
    -------
    widths : dict
        Dictionary containing:
        - "trough_width_us": width of main trough in microseconds
        - "peak_before_width_us": width of main peak before trough in microseconds
        - "peak_after_width_us": width of main peak after trough in microseconds
    """

    def get_main_width(feature_dict):
        if feature_dict["main_idx"] is None:
            return np.nan
        widths = feature_dict.get("widths", np.array([]))
        if len(widths) == 0:
            return np.nan
        main_idx = feature_dict["main_idx"]
        if main_idx < len(widths):
            return widths[main_idx]
        return np.nan

    # Convert from samples to microseconds
    samples_to_us = 1e6 / sampling_frequency

    trough_width = get_main_width(troughs)
    peak_before_width = get_main_width(peaks_before)
    peak_after_width = get_main_width(peaks_after)

    widths = {
        "trough_width_us": trough_width * samples_to_us if not np.isnan(trough_width) else np.nan,
        "peak_before_width_us": peak_before_width * samples_to_us if not np.isnan(peak_before_width) else np.nan,
        "peak_after_width_us": peak_after_width * samples_to_us if not np.isnan(peak_after_width) else np.nan,
    }

    return widths


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
        troughs, _, peaks_after = get_trough_and_peak_idx(template_single)
        trough_idx = troughs["main_loc"]
        peak_idx = peaks_after["main_loc"]
    if trough_idx is None or peak_idx is None:
        return np.nan
    ptv = (peak_idx - trough_idx) / sampling_frequency
    return ptv


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
        troughs, _, peaks_after = get_trough_and_peak_idx(template_single)
        trough_idx = troughs["main_loc"]
        peak_idx = peaks_after["main_loc"]

    if peak_idx is None or peak_idx == 0:
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
        troughs, _, _ = get_trough_and_peak_idx(template_single)
        trough_idx = troughs["main_loc"]

    times = np.arange(template_single.shape[0]) / sampling_frequency

    if trough_idx is None or trough_idx == 0:
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
        _, _, peaks_after = get_trough_and_peak_idx(template_single)
        peak_idx = peaks_after["main_loc"]

    times = np.arange(template_single.shape[0]) / sampling_frequency

    if peak_idx is None or peak_idx == 0:
        return np.nan
    max_idx = int(peak_idx + ((recovery_window_ms / 1000) * sampling_frequency))
    max_idx = np.min([max_idx, template_single.shape[0]])

    res = scipy.stats.linregress(times[peak_idx:max_idx], template_single[peak_idx:max_idx])
    return res.slope


def get_number_of_peaks(template_single, sampling_frequency, troughs, peaks_before, peaks_after, **kwargs):
    """
    Count the total number of peaks (positive) and troughs (negative) in the template.

    Uses the pre-computed peak/trough detection from get_trough_and_peak_idx which
    applies smoothing for more robust detection.

    Parameters
    ----------
    template_single: numpy.ndarray
        The 1D template waveform
    sampling_frequency : float
        The sampling frequency of the template
    troughs : dict
        Trough detection results from get_trough_and_peak_idx
    peaks_before : dict
        Peak before trough results from get_trough_and_peak_idx
    peaks_after : dict
        Peak after trough results from get_trough_and_peak_idx

    Returns
    -------
    num_positive_peaks : int
        The number of positive peaks (peaks_before + peaks_after)
    num_negative_peaks : int
        The number of negative peaks (troughs)
    """
    # Count peaks (positive) from peaks_before and peaks_after
    num_peaks_before = len(peaks_before["indices"])
    num_peaks_after = len(peaks_after["indices"])
    num_positive = num_peaks_before + num_peaks_after

    # Count troughs (negative)
    num_negative = len(troughs["indices"])

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

    theil = TheilSenRegressor(max_iter=1000)
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
    Compute the spatial decay of the template amplitude over distance.

    Can fit either an exponential decay (with offset) or a linear decay model. Channels are first
    filtered by x-distance tolerance from the max channel, then the closest channels
    in y-distance are used for fitting.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - peak_function: the function to use to compute the peak amplitude ("ptp" or "min")
        - min_r2: the minimum r2 to accept the fit
        - linear_fit: bool, if True use linear fit, otherwise exponential fit
        - channel_tolerance: max x-distance (um) from max channel to include channels
        - min_channels_for_fit: minimum number of valid channels required for fitting
        - num_channels_for_fit: number of closest channels to use for fitting
        - normalize_decay: bool, if True normalize amplitudes to max before fitting

    Returns
    -------
    exp_decay_value : float
        The spatial decay slope (decay constant for exp fit, negative slope for linear fit)
    """
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    def exp_decay(x, decay, amp0, offset):
        return amp0 * np.exp(-decay * x) + offset

    def linear_fit_func(x, a, b):
        return a * x + b

    # Extract parameters
    assert "peak_function" in kwargs, "peak_function must be given as kwarg"
    peak_function = kwargs["peak_function"]
    assert "min_r2" in kwargs, "min_r2 must be given as kwarg"
    min_r2 = kwargs["min_r2"]

    use_linear_fit = kwargs.get("linear_fit", False)
    channel_tolerance = kwargs.get("channel_tolerance", None)
    normalize_decay = kwargs.get("normalize_decay", False)

    # Set defaults based on fit type if not specified
    min_channels_for_fit = kwargs.get("min_channels_for_fit")
    if min_channels_for_fit is None:
        min_channels_for_fit = 5 if use_linear_fit else 8

    num_channels_for_fit = kwargs.get("num_channels_for_fit")
    if num_channels_for_fit is None:
        num_channels_for_fit = 6 if use_linear_fit else 10

    # Compute peak amplitudes per channel
    if peak_function == "ptp":
        fun = np.ptp
    elif peak_function == "min":
        fun = np.min
    else:
        fun = np.ptp

    peak_amplitudes = np.abs(fun(template, axis=0))
    max_channel_idx = np.argmax(peak_amplitudes)
    max_channel_location = channel_locations[max_channel_idx]

    # Channel selection based on tolerance (new bombcell-style) or use all channels (old style)
    if channel_tolerance is not None:
        # Calculate x-distances from max channel
        x_dist = np.abs(channel_locations[:, 0] - max_channel_location[0])

        # Find channels within x-distance tolerance
        valid_x_channels = np.argwhere(x_dist <= channel_tolerance).flatten()

        if len(valid_x_channels) < min_channels_for_fit:
            return np.nan

        # Calculate y-distances for channel selection
        y_dist = np.abs(channel_locations[:, 1] - max_channel_location[1])

        # Set y distances to max for channels outside x tolerance (so they won't be selected)
        y_dist_masked = y_dist.copy()
        y_dist_masked[~np.isin(np.arange(len(y_dist)), valid_x_channels)] = y_dist.max() + 1

        # Select the closest channels in y-distance
        use_these_channels = np.argsort(y_dist_masked)[:num_channels_for_fit]

        # Calculate distances from max channel for selected channels
        channel_distances = np.sqrt(
            np.sum(np.square(channel_locations[use_these_channels] - max_channel_location), axis=1)
        )

        # Get amplitudes for selected channels
        spatial_decay_points = np.max(np.abs(template[:, use_these_channels]), axis=0)

        # Sort by distance
        sort_idx = np.argsort(channel_distances)
        channel_distances_sorted = channel_distances[sort_idx]
        peak_amplitudes_sorted = spatial_decay_points[sort_idx]

        # Normalize if requested
        if normalize_decay:
            peak_amplitudes_sorted = peak_amplitudes_sorted / np.max(peak_amplitudes_sorted)

        # Ensure float64 for numerical stability
        channel_distances_sorted = np.float64(channel_distances_sorted)
        peak_amplitudes_sorted = np.float64(peak_amplitudes_sorted)

    else:
        # Old style: use all channels sorted by distance
        channel_distances = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations])
        distances_sort_indices = np.argsort(channel_distances)

        # longdouble is float128 when the platform supports it, otherwise it is float64
        channel_distances_sorted = channel_distances[distances_sort_indices].astype(np.longdouble)
        peak_amplitudes_sorted = peak_amplitudes[distances_sort_indices].astype(np.longdouble)

    try:
        if use_linear_fit:
            # Linear fit: y = a*x + b
            popt, _ = curve_fit(linear_fit_func, channel_distances_sorted, peak_amplitudes_sorted)
            predicted = linear_fit_func(channel_distances_sorted, *popt)
            r2 = r2_score(peak_amplitudes_sorted, predicted)
            exp_decay_value = -popt[0]  # Negative of slope
        else:
            # Exponential fit with offset: y = amp0 * exp(-decay * x) + offset
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

    except Exception:
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


class PeakToTroughDuration(BaseMetric):
    metric_name = "peak_to_trough_duration"
    metric_params = {}
    metric_columns = {"peak_to_trough_duration": float}
    metric_descriptions = {
        "peak_to_trough_duration": "Duration in seconds between the trough (minimum) and the peak (maximum) of the spike waveform."
    }
    needs_tmp_data = True

    @staticmethod
    def _peak_to_trough_duration_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
        return single_channel_metric(
            unit_function=get_peak_to_valley,
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            tmp_data=tmp_data,
            **metric_params,
        )

    metric_function = _peak_to_trough_duration_metric_function


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
    from collections import namedtuple

    num_peaks_result = namedtuple("NumberOfPeaksResult", ["num_positive_peaks", "num_negative_peaks"])
    num_positive_peaks_dict = {}
    num_negative_peaks_dict = {}
    sampling_frequency = tmp_data["sampling_frequency"]
    templates_single = tmp_data["templates_single"]
    troughs_info = tmp_data["troughs_info"]
    peaks_before_info = tmp_data["peaks_before_info"]
    peaks_after_info = tmp_data["peaks_after_info"]
    for unit_index, unit_id in enumerate(unit_ids):
        template_single = templates_single[unit_index]
        num_positive, num_negative = get_number_of_peaks(
            template_single,
            sampling_frequency,
            troughs_info[unit_id],
            peaks_before_info[unit_id],
            peaks_after_info[unit_id],
            **metric_params,
        )
        num_positive_peaks_dict[unit_id] = num_positive
        num_negative_peaks_dict[unit_id] = num_negative
    return num_peaks_result(num_positive_peaks=num_positive_peaks_dict, num_negative_peaks=num_negative_peaks_dict)


class NumberOfPeaks(BaseMetric):
    metric_name = "number_of_peaks"
    metric_function = _number_of_peaks_metric_function
    metric_params = {}
    metric_columns = {"num_positive_peaks": int, "num_negative_peaks": int}
    metric_descriptions = {
        "num_positive_peaks": "Number of positive peaks in the template",
        "num_negative_peaks": "Number of negative peaks (troughs) in the template",
    }
    needs_tmp_data = True


def _main_to_next_peak_duration_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
    result = {}
    templates_single = tmp_data["templates_single"]
    troughs_info = tmp_data["troughs_info"]
    peaks_before_info = tmp_data["peaks_before_info"]
    peaks_after_info = tmp_data["peaks_after_info"]
    sampling_frequency = tmp_data["sampling_frequency"]
    for unit_index, unit_id in enumerate(unit_ids):
        template_single = templates_single[unit_index]
        value = get_main_to_next_peak_duration(
            template_single,
            sampling_frequency,
            troughs_info[unit_id],
            peaks_before_info[unit_id],
            peaks_after_info[unit_id],
            **metric_params,
        )
        result[unit_id] = value
    return result


class MainToNextPeakDuration(BaseMetric):
    metric_name = "main_to_next_peak_duration"
    metric_function = _main_to_next_peak_duration_metric_function
    metric_params = {}
    metric_columns = {"main_to_next_peak_duration": float}
    metric_descriptions = {
        "main_to_next_peak_duration": "Duration in microseconds from main extremum to next extremum."
    }
    needs_tmp_data = True


def _waveform_ratios_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
    from collections import namedtuple

    waveform_ratios_result = namedtuple(
        "WaveformRatiosResult",
        [
            "peak_before_to_trough_ratio",
            "peak_after_to_trough_ratio",
            "peak_before_to_peak_after_ratio",
            "main_peak_to_trough_ratio",
        ],
    )
    peak_before_to_trough = {}
    peak_after_to_trough = {}
    peak_before_to_peak_after = {}
    main_peak_to_trough = {}
    templates_single = tmp_data["templates_single"]
    troughs_info = tmp_data["troughs_info"]
    peaks_before_info = tmp_data["peaks_before_info"]
    peaks_after_info = tmp_data["peaks_after_info"]
    for unit_index, unit_id in enumerate(unit_ids):
        template_single = templates_single[unit_index]
        ratios = get_waveform_ratios(
            template_single,
            troughs_info[unit_id],
            peaks_before_info[unit_id],
            peaks_after_info[unit_id],
            **metric_params,
        )
        peak_before_to_trough[unit_id] = ratios["peak_before_to_trough_ratio"]
        peak_after_to_trough[unit_id] = ratios["peak_after_to_trough_ratio"]
        peak_before_to_peak_after[unit_id] = ratios["peak_before_to_peak_after_ratio"]
        main_peak_to_trough[unit_id] = ratios["main_peak_to_trough_ratio"]
    return waveform_ratios_result(
        peak_before_to_trough_ratio=peak_before_to_trough,
        peak_after_to_trough_ratio=peak_after_to_trough,
        peak_before_to_peak_after_ratio=peak_before_to_peak_after,
        main_peak_to_trough_ratio=main_peak_to_trough,
    )


class WaveformRatios(BaseMetric):
    metric_name = "waveform_ratios"
    metric_function = _waveform_ratios_metric_function
    metric_params = {}
    metric_columns = {
        "peak_before_to_trough_ratio": float,
        "peak_after_to_trough_ratio": float,
        "peak_before_to_peak_after_ratio": float,
        "main_peak_to_trough_ratio": float,
    }
    metric_descriptions = {
        "peak_before_to_trough_ratio": "Ratio of peak before amplitude to trough amplitude",
        "peak_after_to_trough_ratio": "Ratio of peak after amplitude to trough amplitude",
        "peak_before_to_peak_after_ratio": "Ratio of peak before amplitude to peak after amplitude",
        "main_peak_to_trough_ratio": "Ratio of main peak amplitude to trough amplitude",
    }
    needs_tmp_data = True


def _waveform_widths_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
    from collections import namedtuple

    waveform_widths_result = namedtuple(
        "WaveformWidthsResult", ["trough_width", "peak_before_width", "peak_after_width"]
    )
    trough_width_dict = {}
    peak_before_width_dict = {}
    peak_after_width_dict = {}
    templates_single = tmp_data["templates_single"]
    troughs_info = tmp_data["troughs_info"]
    peaks_before_info = tmp_data["peaks_before_info"]
    peaks_after_info = tmp_data["peaks_after_info"]
    sampling_frequency = tmp_data["sampling_frequency"]
    for unit_index, unit_id in enumerate(unit_ids):
        template_single = templates_single[unit_index]
        widths = get_waveform_widths(
            template_single,
            sampling_frequency,
            troughs_info[unit_id],
            peaks_before_info[unit_id],
            peaks_after_info[unit_id],
            **metric_params,
        )
        trough_width_dict[unit_id] = widths["trough_width_us"]
        peak_before_width_dict[unit_id] = widths["peak_before_width_us"]
        peak_after_width_dict[unit_id] = widths["peak_after_width_us"]
    return waveform_widths_result(
        trough_width=trough_width_dict, peak_before_width=peak_before_width_dict, peak_after_width=peak_after_width_dict
    )


class WaveformWidths(BaseMetric):
    metric_name = "waveform_widths"
    metric_function = _waveform_widths_metric_function
    metric_params = {}
    metric_columns = {
        "trough_width": float,
        "peak_before_width": float,
        "peak_after_width": float,
    }
    metric_descriptions = {
        "trough_width": "Width of the main trough in microseconds",
        "peak_before_width": "Width of the main peak before trough in microseconds",
        "peak_after_width": "Width of the main peak after trough in microseconds",
    }
    needs_tmp_data = True


def _waveform_baseline_flatness_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
    result = {}
    templates_single = tmp_data["templates_single"]
    sampling_frequency = tmp_data["sampling_frequency"]
    for unit_index, unit_id in enumerate(unit_ids):
        template_single = templates_single[unit_index]
        value = get_waveform_baseline_flatness(template_single, sampling_frequency, **metric_params)
        result[unit_id] = value
    return result


class WaveformBaselineFlatness(BaseMetric):
    metric_name = "waveform_baseline_flatness"
    metric_function = _waveform_baseline_flatness_metric_function
    metric_params = {"baseline_window_ms": (0.0, 0.5)}
    metric_columns = {"waveform_baseline_flatness": float}
    metric_descriptions = {
        "waveform_baseline_flatness": "Ratio of max baseline amplitude to max waveform amplitude. Lower = flatter baseline."
    }
    needs_tmp_data = True


single_channel_metrics = [
    PeakToTroughDuration,
    HalfWidth,
    RepolarizationSlope,
    RecoverySlope,
    NumberOfPeaks,
    MainToNextPeakDuration,
    WaveformRatios,
    WaveformWidths,
    WaveformBaselineFlatness,
]


def _get_velocity_fits_metric_function(sorting_analyzer, unit_ids, tmp_data, **metric_params):
    from collections import namedtuple

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
    metric_params = {
        "peak_function": "ptp",
        "min_r2": 0.2,
        "linear_fit": False,
        "channel_tolerance": None,  # None uses old style (all channels), set to e.g. 33 for bombcell-style
        "min_channels_for_fit": None,  # None means use default based on linear_fit (5 for linear, 8 for exp)
        "num_channels_for_fit": None,  # None means use default based on linear_fit (6 for linear, 10 for exp)
        "normalize_decay": False,
    }
    metric_columns = {"exp_decay": float}
    metric_descriptions = {
        "exp_decay": (
            "Spatial decay of the template amplitude over distance from the extremum channel (1/um). "
            "Uses exponential or linear fit based on linear_fit parameter."
        )
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
