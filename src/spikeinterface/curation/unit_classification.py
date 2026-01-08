"""
Unit classification based on quality metrics (Bombcell).

Unit Types:
    0 (NOISE): Failed waveform quality checks
    1 (GOOD): Passed all thresholds
    2 (MUA): Failed spike quality checks
    3 (NON_SOMA): Non-somatic units (axonal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


WAVEFORM_METRICS = [
    "num_positive_peaks",
    "num_negative_peaks",
    "peak_to_trough_duration",
    "waveform_baseline_flatness",
    "peak_after_to_trough_ratio",
    "exp_decay",
]

SPIKE_QUALITY_METRICS = [
    "amplitude_median",
    "snr_bombcell",
    "amplitude_cutoff",
    "num_spikes",
    "rp_contamination",
    "presence_ratio",
    "drift_ptp",
]

NON_SOMATIC_METRICS = [
    "peak_before_to_trough_ratio",
    "peak_before_width",
    "trough_width",
    "peak_before_to_peak_after_ratio",
    "main_peak_to_trough_ratio",
]


def bombcell_get_default_thresholds() -> dict: 
    """
    Bombcell - Returns default thresholds for unit classification.

    Each metric has 'min' and 'max' values. Use np.nan to disable a threshold (e.g. to ignore a metric completly
    or to only have a min or a max threshold)
    """
    # bombcell 
    return {
        # Waveform quality (failures -> NOISE)
        "num_positive_peaks": {"min": np.nan, "max": 2},
        "num_negative_peaks": {"min": np.nan, "max": 1},
        "peak_to_trough_duration": {"min": 0.0001, "max": 0.00115},  # seconds
        "waveform_baseline_flatness": {"min": np.nan, "max": 0.5},
        "peak_after_to_trough_ratio": {"min": np.nan, "max": 0.8},
        "exp_decay": {"min": 0.01, "max": 0.1},
        # Spike quality (failures -> MUA)
        "amplitude_median": {"min": 40, "max": np.nan},  # uV
        "snr_bombcell": {"min": 5, "max": np.nan},
        "amplitude_cutoff": {"min": np.nan, "max": 0.2},
        "num_spikes": {"min": 300, "max": np.nan},
        "rp_contamination": {"min": np.nan, "max": 0.1},
        "presence_ratio": {"min": 0.7, "max": np.nan},
        "drift_ptp": {"min": np.nan, "max": 100},  # um
        # Non-somatic detection
        "peak_before_to_trough_ratio": {"min": np.nan, "max": 3},
        "peak_before_width": {"min": 150, "max": np.nan},  # us
        "trough_width": {"min": 200, "max": np.nan},  # us
        "peak_before_to_peak_after_ratio": {"min": np.nan, "max": 3},
        "main_peak_to_trough_ratio": {"min": np.nan, "max": 0.8},
    }


def bombcell_classify_units(
    quality_metrics: pd.DataFrame,
    thresholds: Optional[dict] = None,
    classify_non_somatic: bool = True,
    split_non_somatic_good_mua: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bombcell - classify units based on quality metrics and thresholds.

    Parameters
    ----------
    quality_metrics : pd.DataFrame
        DataFrame with quality metrics (index = unit_ids).
    thresholds : dict or None
        Threshold dict: {"metric": {"min": val, "max": val}}. Use np.nan to disable.
    classify_non_somatic : bool
        If True, detect non-somatic (axonal) units.
    split_non_somatic_good_mua : bool
        If True, split non-somatic into NON_SOMA_GOOD (3) and NON_SOMA_MUA (4).

    Returns
    -------
    unit_type : np.ndarray
        Numeric: 0=NOISE, 1=GOOD, 2=MUA, 3=NON_SOMA
    unit_type_string : np.ndarray
        String labels.
    """
    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()

    n_units = len(quality_metrics)
    unit_type = np.full(n_units, np.nan)
    absolute_value_metrics = ["amplitude_median"]

    # NOISE: waveform failures
    noise_mask = np.zeros(n_units, dtype=bool)
    for metric_name in WAVEFORM_METRICS:
        if metric_name not in quality_metrics.columns or metric_name not in thresholds:
            continue
        values = quality_metrics[metric_name].values
        if metric_name in absolute_value_metrics:
            values = np.abs(values)
        thresh = thresholds[metric_name]
        noise_mask |= np.isnan(values)
        if not np.isnan(thresh["min"]):
            noise_mask |= values < thresh["min"]
        if not np.isnan(thresh["max"]):
            noise_mask |= values > thresh["max"]
    unit_type[noise_mask] = 0

    # MUA: spike quality failures
    mua_mask = np.zeros(n_units, dtype=bool)
    for metric_name in SPIKE_QUALITY_METRICS:
        if metric_name not in quality_metrics.columns or metric_name not in thresholds:
            continue
        values = quality_metrics[metric_name].values
        if metric_name in absolute_value_metrics:
            values = np.abs(values)
        thresh = thresholds[metric_name]
        valid_mask = np.isnan(unit_type)
        if not np.isnan(thresh["min"]):
            mua_mask |= valid_mask & ~np.isnan(values) & (values < thresh["min"])
        if not np.isnan(thresh["max"]):
            mua_mask |= valid_mask & ~np.isnan(values) & (values > thresh["max"])
    unit_type[mua_mask & np.isnan(unit_type)] = 2

    # GOOD: passed all checks
    unit_type[np.isnan(unit_type)] = 1

    # NON-SOMATIC
    if classify_non_somatic:
        def get_metric(name):
            if name in quality_metrics.columns:
                return quality_metrics[name].values
            return np.full(n_units, np.nan)

        peak_before_width = get_metric("peak_before_width")
        trough_width = get_metric("trough_width")
        width_thresh_peak = thresholds.get("peak_before_width", {}).get("min", np.nan)
        width_thresh_trough = thresholds.get("trough_width", {}).get("min", np.nan)

        narrow_peak = (
            ~np.isnan(peak_before_width) & (peak_before_width < width_thresh_peak)
            if not np.isnan(width_thresh_peak)
            else np.zeros(n_units, dtype=bool)
        )
        narrow_trough = (
            ~np.isnan(trough_width) & (trough_width < width_thresh_trough)
            if not np.isnan(width_thresh_trough)
            else np.zeros(n_units, dtype=bool)
        )
        width_conditions = narrow_peak & narrow_trough

        peak_before_to_trough = get_metric("peak_before_to_trough_ratio")
        peak_before_to_peak_after = get_metric("peak_before_to_peak_after_ratio")
        main_peak_to_trough = get_metric("main_peak_to_trough_ratio")

        ratio_thresh_pbt = thresholds.get("peak_before_to_trough_ratio", {}).get("max", np.nan)
        ratio_thresh_pbpa = thresholds.get("peak_before_to_peak_after_ratio", {}).get("max", np.nan)
        ratio_thresh_mpt = thresholds.get("main_peak_to_trough_ratio", {}).get("max", np.nan)

        large_initial_peak = (
            ~np.isnan(peak_before_to_trough) & (peak_before_to_trough > ratio_thresh_pbt)
            if not np.isnan(ratio_thresh_pbt)
            else np.zeros(n_units, dtype=bool)
        )
        large_peak_ratio = (
            ~np.isnan(peak_before_to_peak_after) & (peak_before_to_peak_after > ratio_thresh_pbpa)
            if not np.isnan(ratio_thresh_pbpa)
            else np.zeros(n_units, dtype=bool)
        )
        large_main_peak = (
            ~np.isnan(main_peak_to_trough) & (main_peak_to_trough > ratio_thresh_mpt)
            if not np.isnan(ratio_thresh_mpt)
            else np.zeros(n_units, dtype=bool)
        )

        # (ratio AND width) OR standalone main_peak_to_trough
        ratio_conditions = large_initial_peak | large_peak_ratio
        is_non_somatic = (ratio_conditions & width_conditions) | large_main_peak

        if split_non_somatic_good_mua:
            unit_type[(unit_type == 1) & is_non_somatic] = 3
            unit_type[(unit_type == 2) & is_non_somatic] = 4
        else:
            unit_type[(unit_type != 0) & is_non_somatic] = 3

    # String labels
    if split_non_somatic_good_mua:
        labels = {0: "NOISE", 1: "GOOD", 2: "MUA", 3: "NON_SOMA_GOOD", 4: "NON_SOMA_MUA"}
    else:
        labels = {0: "NOISE", 1: "GOOD", 2: "MUA", 3: "NON_SOMA"}

    unit_type_string = np.array([labels.get(int(t), "UNKNOWN") for t in unit_type], dtype=object)
    return unit_type.astype(int), unit_type_string


def apply_thresholds(
    quality_metrics: pd.DataFrame,
    thresholds: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply thresholds and return pass/fail status for each metric.
    Useful for debugging classification results.
    """
    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()

    results = {}
    for metric_name, thresh in thresholds.items():
        if metric_name not in quality_metrics.columns:
            continue

        values = quality_metrics[metric_name].values
        n_units = len(values)
        passes = np.ones(n_units, dtype=bool)
        reasons = np.array([""] * n_units, dtype=object)

        nan_mask = np.isnan(values)
        passes[nan_mask] = False
        reasons[nan_mask] = "nan"

        if not np.isnan(thresh["min"]):
            below_min = ~nan_mask & (values < thresh["min"])
            passes[below_min] = False
            reasons[below_min] = "below_min"

        if not np.isnan(thresh["max"]):
            above_max = ~nan_mask & (values > thresh["max"])
            passes[above_max] = False
            reasons[above_max & (reasons == "")] = "above_max"
            reasons[above_max & (reasons == "below_min")] = "below_min_and_above_max"

        results[f"{metric_name}_pass"] = passes
        results[f"{metric_name}_fail_reason"] = reasons

    return pd.DataFrame(results, index=quality_metrics.index)


def get_classification_summary(unit_type: np.ndarray, unit_type_string: np.ndarray) -> dict:
    """Get counts and percentages for each unit type."""
    n_total = len(unit_type)
    unique_types, counts = np.unique(unit_type, return_counts=True)

    summary = {"total_units": n_total, "counts": {}, "percentages": {}}
    for utype, count in zip(unique_types, counts):
        label = unit_type_string[unit_type == utype][0]
        summary["counts"][label] = int(count)
        summary["percentages"][label] = round(100 * count / n_total, 1)

    return summary


def save_thresholds(thresholds: dict, filepath) -> None:
    """
    Save thresholds to a JSON file.

    Parameters
    ----------
    thresholds : dict
        Threshold dictionary from bombcell_get_default_thresholds() or modified version.
    filepath : str or Path
        Path to save the JSON file.
    """
    import json
    from pathlib import Path

    # Convert np.nan to None for JSON serialization
    json_thresholds = {}
    for metric_name, thresh in thresholds.items():
        json_thresholds[metric_name] = {
            "min": None if (isinstance(thresh["min"], float) and np.isnan(thresh["min"])) else thresh["min"],
            "max": None if (isinstance(thresh["max"], float) and np.isnan(thresh["max"])) else thresh["max"],
        }

    filepath = Path(filepath)
    with open(filepath, "w") as f:
        json.dump(json_thresholds, f, indent=4)


def load_thresholds(filepath) -> dict:
    """
    Load thresholds from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file.

    Returns
    -------
    thresholds : dict
        Threshold dictionary compatible with bombcell_classify_units().
    """
    import json
    from pathlib import Path

    filepath = Path(filepath)
    with open(filepath, "r") as f:
        json_thresholds = json.load(f)

    # Convert None to np.nan
    thresholds = {}
    for metric_name, thresh in json_thresholds.items():
        thresholds[metric_name] = {
            "min": np.nan if thresh["min"] is None else thresh["min"],
            "max": np.nan if thresh["max"] is None else thresh["max"],
        }

    return thresholds
