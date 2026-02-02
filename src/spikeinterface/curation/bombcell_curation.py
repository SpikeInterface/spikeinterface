"""
Unit labeling based on quality metrics (bombcell).

Unit Types:
    0 (NOISE): Failed waveform quality checks
    1 (GOOD): Passed all thresholds
    2 (MUA): Failed spike quality checks
    3 (NON_SOMA): Non-somatic units (axonal)
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

NOISE_METRICS = [
    "num_positive_peaks",
    "num_negative_peaks",
    "peak_to_trough_duration",
    "waveform_baseline_flatness",
    "peak_after_to_trough_ratio",
    "exp_decay",
]

SPIKE_QUALITY_METRICS = [
    "amplitude_median",
    "snr",
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
    bombcell - Returns default thresholds for unit labeling.

    Each metric has 'min' and 'max' values. Use None to disable a threshold (e.g. to ignore a metric completely
    or to only have a min or a max threshold)
    """
    # bombcell
    return {
        # Waveform quality (failures -> NOISE)
        "num_positive_peaks": {"min": None, "max": 2},
        "num_negative_peaks": {"min": None, "max": 1},
        "peak_to_trough_duration": {"min": 0.0001, "max": 0.00115},  # seconds
        "waveform_baseline_flatness": {"min": None, "max": 0.5},
        "peak_after_to_trough_ratio": {"min": None, "max": 0.8},
        "exp_decay": {"min": 0.01, "max": 0.1},
        # Spike quality (failures -> MUA)
        "amplitude_median": {"min": 40, "max": None},  # uV
        "snr_baseline": {"min": 5, "max": None},
        "amplitude_cutoff": {"min": None, "max": 0.2},
        "num_spikes": {"min": 300, "max": None},
        "rp_contamination": {"min": None, "max": 0.1},
        "presence_ratio": {"min": 0.7, "max": None},
        "drift_ptp": {"min": None, "max": 100},  # um
        # Non-somatic detection
        "peak_before_to_trough_ratio": {"min": None, "max": 3},
        "peak_before_width": {"min": 0.00015, "max": None},  # seconds
        "trough_width": {"min": 0.0002, "max": None},  # seconds
        "peak_before_to_peak_after_ratio": {"min": None, "max": 3},
        "main_peak_to_trough_ratio": {"min": None, "max": 0.8},
    }


def _is_threshold_disabled(value):
    """Check if a threshold value is disabled (None or np.nan)."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def bombcell_label_units(
    sorting_analyzer=None,
    thresholds: dict | str | Path | None = None,
    label_non_somatic: bool = True,
    split_non_somatic_good_mua: bool = False,
    external_metrics: "pd.DataFrame | list[pd.DataFrame]" | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    bombcell - label units based on quality metrics and thresholds.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer, optional
        SortingAnalyzer with computed quality_metrics and/or template_metrics extensions.
        If provided, metrics are extracted automatically using get_metrics_extension_data().
    thresholds : dict | str | Path | None
        Threshold dict or JSON file, including a set of {"metric": {"min": val, "max": val}}.
        If None, default bombcell thresholds are used.
    label_non_somatic : bool
        If True, detect non-somatic (axonal) units.
    split_non_somatic_good_mua : bool
        If True, split non-somatic into NON_SOMA_GOOD (3) and NON_SOMA_MUA (4).
    external_metrics: "pd.DataFrame | list[pd.DataFrame]" | None = None
        External metrics DataFrame(s) (index = unit_ids) to use instead of those from SortingAnalyzer.

    Returns
    -------
    unit_type : np.ndarray
        Numeric: 0=NOISE, 1=GOOD, 2=MUA, 3=NON_SOMA
    unit_type_string : np.ndarray
        String labels.
    """
    import pandas as pd

    if sorting_analyzer is not None:
        combined_metrics = sorting_analyzer.get_metrics_extension_data()
        if combined_metrics.empty:
            raise ValueError(
                "SortingAnalyzer has no metrics extensions computed. "
                "Compute quality_metrics and/or template_metrics first."
            )
    else:
        if external_metrics is None:
            raise ValueError("Either sorting_analyzer or external_metrics must be provided")
        if isinstance(external_metrics, list):
            assert all(
                isinstance(df, pd.DataFrame) for df in external_metrics
            ), "All items in external_metrics must be DataFrames"
            combined_metrics = pd.concat(external_metrics, axis=1)
        else:
            combined_metrics = external_metrics

    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()
    elif isinstance(thresholds, (str, Path)):
        with open(thresholds, "r") as f:
            thresholds = json.load(f)
    elif not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict, a JSON file path, or None")

    n_units = len(combined_metrics)
    unit_type = np.full(n_units, np.nan)
    absolute_value_metrics = ["amplitude_median"]

    # NOISE: waveform failures
    noise_mask = np.zeros(n_units, dtype=bool)
    for metric_name in NOISE_METRICS:
        if metric_name not in combined_metrics.columns or metric_name not in thresholds:
            continue
        values = combined_metrics[metric_name].values
        if metric_name in absolute_value_metrics:
            values = np.abs(values)
        thresh = thresholds[metric_name]
        noise_mask |= np.isnan(values)
        if not _is_threshold_disabled(thresh["min"]):
            noise_mask |= values < thresh["min"]
        if not _is_threshold_disabled(thresh["max"]):
            noise_mask |= values > thresh["max"]
    unit_type[noise_mask] = 0

    # MUA: spike quality failures
    mua_mask = np.zeros(n_units, dtype=bool)
    for metric_name in SPIKE_QUALITY_METRICS:
        if metric_name not in combined_metrics.columns or metric_name not in thresholds:
            continue
        values = combined_metrics[metric_name].values
        if metric_name in absolute_value_metrics:
            values = np.abs(values)
        thresh = thresholds[metric_name]
        valid_mask = np.isnan(unit_type)
        if not _is_threshold_disabled(thresh["min"]):
            mua_mask |= valid_mask & ~np.isnan(values) & (values < thresh["min"])
        if not _is_threshold_disabled(thresh["max"]):
            mua_mask |= valid_mask & ~np.isnan(values) & (values > thresh["max"])
    unit_type[mua_mask & np.isnan(unit_type)] = 2

    # GOOD: passed all checks
    unit_type[np.isnan(unit_type)] = 1

    # NON-SOMATIC
    if label_non_somatic:

        def get_metric(name):
            if name in combined_metrics.columns:
                return combined_metrics[name].values
            return np.full(n_units, np.nan)

        peak_before_width = get_metric("peak_before_width")
        trough_width = get_metric("trough_width")
        width_thresh_peak = thresholds.get("peak_before_width", {}).get("min", None)
        width_thresh_trough = thresholds.get("trough_width", {}).get("min", None)

        narrow_peak = (
            ~np.isnan(peak_before_width) & (peak_before_width < width_thresh_peak)
            if not _is_threshold_disabled(width_thresh_peak)
            else np.zeros(n_units, dtype=bool)
        )
        narrow_trough = (
            ~np.isnan(trough_width) & (trough_width < width_thresh_trough)
            if not _is_threshold_disabled(width_thresh_trough)
            else np.zeros(n_units, dtype=bool)
        )
        width_conditions = narrow_peak & narrow_trough

        peak_before_to_trough = get_metric("peak_before_to_trough_ratio")
        peak_before_to_peak_after = get_metric("peak_before_to_peak_after_ratio")
        main_peak_to_trough = get_metric("main_peak_to_trough_ratio")

        ratio_thresh_pbt = thresholds.get("peak_before_to_trough_ratio", {}).get("max", None)
        ratio_thresh_pbpa = thresholds.get("peak_before_to_peak_after_ratio", {}).get("max", None)
        ratio_thresh_mpt = thresholds.get("main_peak_to_trough_ratio", {}).get("max", None)

        large_initial_peak = (
            ~np.isnan(peak_before_to_trough) & (peak_before_to_trough > ratio_thresh_pbt)
            if not _is_threshold_disabled(ratio_thresh_pbt)
            else np.zeros(n_units, dtype=bool)
        )
        large_peak_ratio = (
            ~np.isnan(peak_before_to_peak_after) & (peak_before_to_peak_after > ratio_thresh_pbpa)
            if not _is_threshold_disabled(ratio_thresh_pbpa)
            else np.zeros(n_units, dtype=bool)
        )
        large_main_peak = (
            ~np.isnan(main_peak_to_trough) & (main_peak_to_trough > ratio_thresh_mpt)
            if not _is_threshold_disabled(ratio_thresh_mpt)
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
        labels = {0: "NOISE", 1: "good", 2: "mua", 3: "non_soma_good", 4: "non_soma_mua"}
    else:
        labels = {0: "noise", 1: "good", 2: "mua", 3: "non_soma"}

    unit_type_string = np.array([labels.get(int(t), "unknown") for t in unit_type], dtype=object)
    return unit_type.astype(int), unit_type_string


def get_bombcell_labeling_summary(unit_type: np.ndarray, unit_type_string: np.ndarray) -> dict:
    """Get counts and percentages for each unit type."""
    n_total = len(unit_type)
    unique_types, counts = np.unique(unit_type, return_counts=True)

    summary = {"total_units": n_total, "counts": {}, "percentages": {}}
    for utype, count in zip(unique_types, counts):
        label = unit_type_string[unit_type == utype][0]
        summary["counts"][label] = int(count)
        summary["percentages"][label] = round(100 * count / n_total, 1)

    return summary


def save_bombcell_results(
    quality_metrics: "pd.DataFrame",
    unit_type: np.ndarray,
    unit_type_string: np.ndarray,
    thresholds: dict,
    folder,
    save_narrow: bool = True,
    save_wide: bool = True,
) -> None:
    """
    Save labeling results to CSV files.

    Parameters
    ----------
    quality_metrics : pd.DataFrame
        DataFrame with quality metrics (index = unit_ids).
    unit_type : np.ndarray
        Numeric unit type codes.
    unit_type_string : np.ndarray
        String labels for each unit.
    thresholds : dict
        Threshold dictionary used for labeling.
    folder : str or Path
        Folder to save the CSV files.
    save_narrow : bool, default: True
        Save narrow/tidy format (one row per unit-metric).
    save_wide : bool, default: True
        Save wide format (one row per unit, metrics as columns).
    """
    from pathlib import Path
    import pandas as pd

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    unit_ids = quality_metrics.index.values

    # Wide format: one row per unit
    if save_wide:
        wide_df = quality_metrics.copy()
        wide_df.insert(0, "label", unit_type_string)
        wide_df.insert(1, "label_code", unit_type)
        wide_df.to_csv(folder / "labeling_results_wide.csv")

    # Narrow format: one row per unit-metric combination
    if save_narrow:
        rows = []
        for i, unit_id in enumerate(unit_ids):
            label = unit_type_string[i]
            label_code = unit_type[i]
            for metric_name in quality_metrics.columns:
                if metric_name not in thresholds:
                    continue
                value = quality_metrics.loc[unit_id, metric_name]
                thresh = thresholds[metric_name]
                thresh_min = thresh.get("min", None)
                thresh_max = thresh.get("max", None)

                # Determine pass/fail
                passed = True
                if np.isnan(value):
                    passed = False
                elif not _is_threshold_disabled(thresh_min) and value < thresh_min:
                    passed = False
                elif not _is_threshold_disabled(thresh_max) and value > thresh_max:
                    passed = False

                rows.append(
                    {
                        "unit_id": unit_id,
                        "label": label,
                        "label_code": label_code,
                        "metric_name": metric_name,
                        "value": value,
                        "threshold_min": thresh_min,
                        "threshold_max": thresh_max,
                        "passed": passed,
                    }
                )

        narrow_df = pd.DataFrame(rows)
        narrow_df.to_csv(folder / "labeling_results_narrow.csv", index=False)
