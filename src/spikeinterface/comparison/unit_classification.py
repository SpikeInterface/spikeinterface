"""
Unit classification based on quality metrics and user-defined thresholds.

This module provides functionality to classify neural units based on quality metrics
(similar to BombCell). Each metric can have min and max thresholds - use NaN to
disable a threshold.

Unit Types:
    0 (NOISE): Units failing waveform quality checks
    1 (GOOD): Units passing all quality thresholds
    2 (MUA): Multi-unit activity - units failing spike quality checks but not waveform checks
    3 (NON_SOMA): Non-somatic units (axonal, etc.) - optional classification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def get_default_thresholds() -> dict:
    """
    Returns default thresholds for unit classification.

    Each threshold entry has 'min' and 'max' values. Use np.nan to disable
    a threshold direction (e.g., if only a minimum matters, set max to np.nan).

    Thresholds are organized by category:
    - waveform: Template/waveform shape checks (failures -> NOISE)
    - spike_quality: Spike sorting quality checks (failures -> MUA)
    - non_somatic: Non-somatic detection (optional, failures -> NON_SOMA)

    Returns
    -------
    thresholds : dict
        Dictionary of threshold parameters with min/max values.

    Notes
    -----
    Metric names correspond to SpikeInterface metric column names:

    Template metrics (from template_metrics extension):
    - num_positive_peaks: Number of positive peaks (repolarization peaks)
    - num_negative_peaks: Number of negative peaks (troughs)
    - waveform_duration: Duration in microseconds
    - waveform_baseline_flatness: Baseline flatness metric
    - peak_after_to_trough_ratio: Ratio of peak after trough to trough amplitude
    - exp_decay: Exponential decay constant for spatial spread

    Quality metrics (from quality_metrics extension):
    - amplitude_median: Median spike amplitude
    - snr: Signal-to-noise ratio
    - amplitude_cutoff: Estimated fraction of missing spikes
    - num_spikes: Total spike count
    - rp_contamination: Refractory period contamination
    - presence_ratio: Fraction of recording where unit is present
    - drift_ptp: Peak-to-peak drift in um
    """
    thresholds = {
        # ============================================================
        # WAVEFORM QUALITY THRESHOLDS (failures classify as NOISE)
        # ============================================================

        # Number of positive peaks (repolarization peaks after trough)
        # Good units typically have 1-2 peaks
        "num_positive_peaks": {"min": np.nan, "max": 2},

        # Number of negative peaks (troughs) in waveform
        # Good units typically have 1 main trough
        "num_negative_peaks": {"min": np.nan, "max": 1},

        # Waveform duration in MICROSECONDS (from template_metrics)
        # Typical range: 100-1150 us
        "waveform_duration": {"min": 100, "max": 1150},

        # Baseline flatness - max deviation as fraction of peak amplitude
        # Lower is better, typical threshold 0.3
        "waveform_baseline_flatness": {"min": np.nan, "max": 0.3},

        # Peak after trough to trough ratio - helps detect noise
        # High values indicate noise (ratio > 0.8 is suspicious)
        "peak_after_to_trough_ratio": {"min": np.nan, "max": 0.8},

        # Exponential decay constant for spatial spread
        # Values outside typical range indicate noise
        "exp_decay": {"min": 0.01, "max": 0.1},

        # ============================================================
        # SPIKE QUALITY THRESHOLDS (failures classify as MUA)
        # ============================================================

        # Median spike amplitude (in uV typically)
        # Lower bound ensures sufficient signal
        "amplitude_median": {"min": 40, "max": np.nan},

        # Signal-to-noise ratio
        # Higher is better, minimum ensures reliable detection
        "snr": {"min": 5, "max": np.nan},

        # Amplitude cutoff - estimates fraction of missing spikes
        # Lower is better (less missing), max 0.2 means <20% estimated missing
        "amplitude_cutoff": {"min": np.nan, "max": 0.2},

        # Minimum number of spikes
        # Ensures sufficient data for reliable metrics
        "num_spikes": {"min": 300, "max": np.nan},

        # Refractory period contamination rate
        # Lower is better, max typically 0.1 (10%)
        "rp_contamination": {"min": np.nan, "max": 0.1},

        # Presence ratio - fraction of recording where unit is active
        # Higher is better, ensures unit present throughout
        "presence_ratio": {"min": 0.7, "max": np.nan},

        # Drift MAD - median absolute deviation of drift in um
        # Lower is better, ensures stable unit location
        "drift_ptp": {"min": np.nan, "max": 100},

        # ============================================================
        # NON-SOMATIC DETECTION THRESHOLDS (optional)
        # ============================================================

        # These thresholds identify axonal/dendritic units by their waveform shape
        # Non-somatic units have characteristic triphasic waveforms

        # Peak before to trough ratio - non-somatic have large initial peak
        "peak_before_to_trough_ratio": {"min": np.nan, "max": 3},  # non-somatic if > max

        # Peak before width (in samples at sampling rate) #QQ should be microseconds or something! 
        "peak_before_width": {"min": 4, "max": np.nan},  # non-somatic if < min

        # Trough width (in samples) #QQ should be microseconds or something! 
        "trough_width": {"min": 5, "max": np.nan},  # non-somatic if < min

        # Peak before to peak after ratio
        "peak_before_to_peak_after_ratio": {"min": np.nan, "max": 3},  # non-somatic if > max

        # Main peak to trough ratio
        "main_peak_to_trough_ratio": {"min": np.nan, "max": 0.8},  # non-somatic if > max
    }

    return thresholds


def classify_units(
    quality_metrics: pd.DataFrame,
    thresholds: Optional[dict] = None,
    classify_non_somatic: bool = False,
    split_non_somatic_good_mua: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify units based on quality metrics and thresholds.

    Classification hierarchy:
    1. NOISE (0): Units failing waveform quality checks
    2. MUA (2): Units passing waveform checks but failing spike quality checks
    3. GOOD (1): Units passing all checks
    4. NON_SOMA (3/4): Optional - units with non-somatic waveform characteristics

    Parameters
    ----------
    quality_metrics : pd.DataFrame
        DataFrame with quality metrics. Index should be unit_ids.
        Can contain metrics from quality_metrics, template_metrics,
        and spiketrain_metrics extensions.
    thresholds : dict or None, default: None
        Threshold dictionary with format {"metric_name": {"min": val, "max": val}}.
        Use np.nan to disable a threshold. If None, uses get_default_thresholds().
    classify_non_somatic : bool, default: False
        If True, also classify non-somatic (axonal) units.
    split_non_somatic_good_mua : bool, default: False
        If True and classify_non_somatic is True, split non-somatic into
        NON_SOMA_GOOD (3) and NON_SOMA_MUA (4). Only applies if
        classify_non_somatic is True.

    Returns
    -------
    unit_type : np.ndarray
        Numeric classification: 0=NOISE, 1=GOOD, 2=MUA, 3=NON_SOMA (or NON_SOMA_GOOD),
        4=NON_SOMA_MUA (if split_non_somatic_good_mua=True)
    unit_type_string : np.ndarray
        String labels for each unit type.

    """
    if thresholds is None:
        thresholds = get_default_thresholds()

    n_units = len(quality_metrics)
    unit_type = np.full(n_units, np.nan)

    # Define which metrics go to which category
    waveform_metrics = [
        "num_positive_peaks",
        "num_negative_peaks",
        "waveform_duration",
        "waveform_baseline_flatness",
        "peak_after_to_trough_ratio",
        "exp_decay",
    ]

    spike_quality_metrics = [
        "amplitude_median",
        "snr",
        "amplitude_cutoff",
        "num_spikes",
        "rp_contamination",
        "presence_ratio",
        "drift_ptp",
    ]

    non_somatic_metrics = [
        "peak_before_to_trough_ratio",
        "peak_before_width",
        "trough_width",
        "peak_before_to_peak_after_ratio",
        "main_peak_to_trough_ratio",
    ]

    # ========================================
    # NOISE classification
    # ========================================
    noise_mask = np.zeros(n_units, dtype=bool)

    for metric_name in waveform_metrics:
        if metric_name not in quality_metrics.columns:
            continue
        if metric_name not in thresholds:
            continue

        values = quality_metrics[metric_name].values
        thresh = thresholds[metric_name]

        # NaN values in metrics are considered failures for waveform metrics
        noise_mask |= np.isnan(values)

        # Check min threshold
        if not np.isnan(thresh["min"]):
            noise_mask |= values < thresh["min"]

        # Check max threshold
        if not np.isnan(thresh["max"]):
            noise_mask |= values > thresh["max"]

    unit_type[noise_mask] = 0

    # ========================================
    # MUA classification
    # ========================================
    mua_mask = np.zeros(n_units, dtype=bool)

    for metric_name in spike_quality_metrics:
        if metric_name not in quality_metrics.columns:
            continue
        if metric_name not in thresholds:
            continue

        values = quality_metrics[metric_name].values
        thresh = thresholds[metric_name]

        # Only apply to units not yet classified as noise
        valid_mask = np.isnan(unit_type)

        # Check min threshold (NaN values don't fail min threshold for spike quality)
        if not np.isnan(thresh["min"]):
            mua_mask |= valid_mask & ~np.isnan(values) & (values < thresh["min"])

        # Check max threshold (NaN values don't fail max threshold for spike quality)
        if not np.isnan(thresh["max"]):
            mua_mask |= valid_mask & ~np.isnan(values) & (values > thresh["max"])

    unit_type[mua_mask & np.isnan(unit_type)] = 2

    # ========================================
    # GOOD classification (passed all checks)
    # ========================================
    unit_type[np.isnan(unit_type)] = 1

    # ========================================
    # NON-SOMATIC classification
    # ========================================
    if classify_non_somatic:
        is_non_somatic = np.zeros(n_units, dtype=bool)

        for metric_name in non_somatic_metrics:
            if metric_name not in quality_metrics.columns:
                continue
            if metric_name not in thresholds:
                continue

            values = quality_metrics[metric_name].values
            thresh = thresholds[metric_name]

            # Non-somatic detection uses OPPOSITE logic:
            # - Values BELOW min threshold -> non-somatic
            # - Values ABOVE max threshold -> non-somatic
            if not np.isnan(thresh["min"]):
                is_non_somatic |= ~np.isnan(values) & (values < thresh["min"])

            if not np.isnan(thresh["max"]):
                is_non_somatic |= ~np.isnan(values) & (values > thresh["max"])

        # Apply non-somatic classification
        if split_non_somatic_good_mua:
            # Split into NON_SOMA_GOOD (3) and NON_SOMA_MUA (4)
            good_non_somatic = (unit_type == 1) & is_non_somatic
            mua_non_somatic = (unit_type == 2) & is_non_somatic
            unit_type[good_non_somatic] = 3
            unit_type[mua_non_somatic] = 4
        else:
            # All non-noise non-somatic units get type 3
            unit_type[(unit_type != 0) & is_non_somatic] = 3

    # ========================================
    # Create string labels
    # ========================================
    if split_non_somatic_good_mua:
        labels = {
            0: "NOISE",
            1: "GOOD",
            2: "MUA",
            3: "NON_SOMA_GOOD",
            4: "NON_SOMA_MUA",
        }
    else:
        labels = {
            0: "NOISE",
            1: "GOOD",
            2: "MUA",
            3: "NON_SOMA",
        }

    unit_type_string = np.array([labels.get(int(t), "UNKNOWN") for t in unit_type], dtype=object)

    return unit_type.astype(int), unit_type_string


def apply_thresholds(
    quality_metrics: pd.DataFrame,
    thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply thresholds to quality metrics and return pass/fail status for each.

    This is useful for debugging which metrics are causing units to fail.

    Parameters
    ----------
    quality_metrics : pd.DataFrame
        DataFrame with quality metrics.
    thresholds : dict or None, default: None
        Threshold dictionary. If None, uses get_default_thresholds().

    Returns
    -------
    threshold_results : pd.DataFrame
        DataFrame with same index as quality_metrics, with columns:
        - {metric}_pass: bool, True if metric passes threshold
        - {metric}_fail_reason: str, reason for failure ("below_min", "above_max", "nan", or "")
    """
    if thresholds is None:
        thresholds = get_default_thresholds()

    results = {}

    for metric_name, thresh in thresholds.items():
        if metric_name not in quality_metrics.columns:
            continue

        values = quality_metrics[metric_name].values
        n_units = len(values)

        # Initialize
        passes = np.ones(n_units, dtype=bool)
        reasons = np.array([""] * n_units, dtype=object)

        # Check for NaN
        nan_mask = np.isnan(values)
        passes[nan_mask] = False
        reasons[nan_mask] = "nan"

        # Check min threshold
        if not np.isnan(thresh["min"]):
            below_min = ~nan_mask & (values < thresh["min"])
            passes[below_min] = False
            reasons[below_min] = "below_min"

        # Check max threshold
        if not np.isnan(thresh["max"]):
            above_max = ~nan_mask & (values > thresh["max"])
            passes[above_max] = False
            # Only overwrite if not already failed
            reasons[above_max & (reasons == "")] = "above_max"
            # If both fail, indicate both
            reasons[above_max & (reasons == "below_min")] = "below_min_and_above_max"

        results[f"{metric_name}_pass"] = passes
        results[f"{metric_name}_fail_reason"] = reasons

    return pd.DataFrame(results, index=quality_metrics.index)


def get_classification_summary(
    unit_type: np.ndarray,
    unit_type_string: np.ndarray,
) -> dict:
    """
    Get summary statistics of unit classification.

    Parameters
    ----------
    unit_type : np.ndarray
        Numeric unit type array from classify_units().
    unit_type_string : np.ndarray
        String labels from classify_units().

    Returns
    -------
    summary : dict
        Dictionary with counts and percentages for each unit type.
    """
    n_total = len(unit_type)
    unique_types, counts = np.unique(unit_type, return_counts=True)

    summary = {
        "total_units": n_total,
        "counts": {},
        "percentages": {},
    }

    # Get the label for each type
    for utype, count in zip(unique_types, counts):
        label = unit_type_string[unit_type == utype][0]
        summary["counts"][label] = int(count)
        summary["percentages"][label] = round(100 * count / n_total, 1)

    return summary
