"""
Unit labeling based on quality metrics (bombcell).

Unit Labels:
    noise: Failed waveform quality checks
    good: Passed all thresholds
    mua: Failed spike quality checks
    non_soma: Non-somatic units (axonal)
"""

from __future__ import annotations

import operator
from pathlib import Path
import json
import numpy as np

from .curation_tools import is_threshold_disabled
from .threshold_metrics_curation import threshold_metrics_label_units

DEFAULT_NOISE_METRICS = [
    "num_positive_peaks",
    "num_negative_peaks",
    "peak_to_trough_duration",
    "waveform_baseline_flatness",
    "peak_after_to_trough_ratio",
    "exp_decay",
]

DEFAULT_MUA_METRICS = [
    "amplitude_median",
    "snr",
    "amplitude_cutoff",
    "num_spikes",
    "rp_contamination",
    "presence_ratio",
    "drift_ptp",
]

DEFAULT_NON_SOMATIC_METRICS = [
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
        "noise": {  # failures -> NOISE
            "num_positive_peaks": {"min": None, "max": 2},
            "num_negative_peaks": {"min": None, "max": 1},
            "peak_to_trough_duration": {"min": 0.0001, "max": 0.00115},  # seconds
            "waveform_baseline_flatness": {"min": None, "max": 0.5},
            "peak_after_to_trough_ratio": {"min": None, "max": 0.8},
            "exp_decay": {"min": 0.01, "max": 0.1},
        },
        "mua": {  # failures -> MUA, only applied to units that passed noise thresholds
            "amplitude_median": {"min": 40, "max": None},  # uV
            "snr": {"min": 5, "max": None},
            "amplitude_cutoff": {"min": None, "max": 0.2},
            "num_spikes": {"min": 300, "max": None},
            "rp_contamination": {"min": None, "max": 0.1},
            "presence_ratio": {"min": 0.7, "max": None},
            "drift_ptp": {"min": None, "max": 100},  # um
        },
        "non-somatic": {
            "peak_before_to_trough_ratio": {"min": None, "max": 3},
            "peak_before_width": {"min": 0.00015, "max": None},  # seconds
            "trough_width": {"min": 0.0002, "max": None},  # seconds
            "peak_before_to_peak_after_ratio": {"min": None, "max": 3},
            "main_peak_to_trough_ratio": {"min": None, "max": 0.8},
        },
    }


def bombcell_label_units(
    sorting_analyzer=None,
    thresholds: dict | str | Path | None = None,
    label_non_somatic: bool = True,
    split_non_somatic_good_mua: bool = False,
    external_metrics: "pd.DataFrame | list[pd.DataFrame]" | None = None,
    implementation: str = "new",
) -> "pd.DataFrame":
    """
    Label units based on quality metrics and template metrics using Bombcell logic:

    1. NOISE:
        Units that fail any of the noise thresholds are labeled as "noise". The thresholds in the "noise" section
        are applied with an "AND" operator, meaning a unit must fail at least one noise metric to be labeled as
        "noise". Units that are not labeled as "noise" are then evaluated for MUA thresholds.
    2. MUA:
        Units that are not "noise" but fail any of the MUA thresholds are labeled as "mua". The thresholds in the
        "mua" section are also applied with an "AND" operator, meaning a unit must fail at least one MUA metric to
        be labeled as "mua".
    3. GOOD:
        Units that pass all noise and MUA thresholds are labeled as "good".
    4. NON-SOMATIC:
        Among units that are not "noise", those that meet non-somatic criteria based on waveform shape are
        labeled as "non_soma".
        These non-somatic criteria include:

        - Narrow peak and trough widths (using "peak_before_width" and "trough_width" metrics)
        - Large ratios of peak_before the trough and/or peak_before to peak_after
          (using "peak_before_to_trough_ratio" and "peak_before_to_peak_after_ratio" metrics)
        - Large main peak to trough ratio (using "main_peak_to_trough_ratio" metric)

        If units have a narrow peak and a large ratio OR a large main peak to trough ratio,
        they are labeled as non-somatic. If `split_non_somatic_good_mua` is True, non-somatic units are further split
        into "non_soma_good" and "non_soma_mua", otherwise they are all labeled as "non_soma".

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer, optional
        SortingAnalyzer with computed quality_metrics and/or template_metrics extensions.
        If provided, metrics are extracted automatically using get_metrics_extension_data().
    thresholds : dict | str | Path | None
        Threshold dict or JSON file, including a three sections ("noise", "mua", "non-somatic") of
        {"metric": {"min": val, "max": val}}.
        If None, default Bombcell thresholds are used.
    label_non_somatic : bool, default: True
        If True, detect non-somatic (dendritic, axonal) units.
    split_non_somatic_good_mua : bool, default: False
        If True, split non-somatic into "non_soma_good" and "non_soma_mua".
    external_metrics: "pd.DataFrame | list[pd.DataFrame]" | None = None
        External metrics DataFrame(s) (index = unit_ids) to use instead of those from SortingAnalyzer.

    Returns
    -------
    labels : pd.DataFrame
        A DataFrame with unit ids as index and "label" as column

    References
    ----------
    Ported by Julie Fabre and Alessio Buccino from the Bombcell repository:
    https://github.com/Julie-Fabre/bombcell
    See [Fabre]_ for more details on the original implementation and rationale behind the thresholds.
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
        thresholds_dict = bombcell_get_default_thresholds()
    elif isinstance(thresholds, (str, Path)):
        with open(thresholds, "r") as f:
            thresholds_dict = json.load(f)
    elif isinstance(thresholds, dict):
        thresholds_dict = thresholds
    else:
        raise ValueError("thresholds must be a dict, a JSON file path, or None")

    n_units = len(combined_metrics)
    absolute_value_metrics = ["amplitude_median"]
    for metric in absolute_value_metrics:
        if metric in combined_metrics.columns:
            combined_metrics[metric] = np.abs(combined_metrics[metric])

    # NOISE: waveform failures
    if implementation == "new":
        noise_thresholds = thresholds_dict.get("noise", {})
        if len(noise_thresholds) > 0:
            unit_labels = threshold_metrics_label_units(
                sorting_analyzer_or_metrics=combined_metrics,
                thresholds=noise_thresholds,
                pass_label="good",
                fail_label="noise",
                operator="and",
                nan_policy="fail",
            )
            (non_noise_indices,) = np.nonzero(unit_labels["label"] == "good")
        else:
            unit_labels = pd.DataFrame(data={"label": np.array(["good"] * n_units)}, index=combined_metrics.index)
            non_noise_indices = np.arange(n_units)
        mua_thresholds = thresholds_dict.get("mua", {})
        if len(mua_thresholds) > 0:
            neural_metrics = combined_metrics.iloc[non_noise_indices]
            mua_labels = threshold_metrics_label_units(
                sorting_analyzer_or_metrics=neural_metrics,
                thresholds=mua_thresholds,
                pass_label="good",
                fail_label="mua",
                operator="and",
                nan_policy="ignore",
            )
            unit_labels.loc[unit_labels.index[non_noise_indices], "label"] = mua_labels["label"].values

        if label_non_somatic:
            non_somatic_thresholds = thresholds_dict.get("non-somatic", {})
            width_thresholds = {
                m: non_somatic_thresholds[m]
                for m in ["peak_before_width", "trough_width"]
                if m in non_somatic_thresholds
            }
            if len(width_thresholds) > 0:
                width_condition_labels = threshold_metrics_label_units(
                    sorting_analyzer_or_metrics=combined_metrics,
                    thresholds=width_thresholds,
                    pass_label="not_narrow_width",
                    fail_label="narrow_width",
                    operator="or",
                    nan_policy="ignore",
                )
            else:
                width_condition_labels = pd.DataFrame(
                    data={"label": np.array(["not_narrow_width"] * len(combined_metrics))}, index=combined_metrics.index
                )

            ratio_thresholds = {
                m: non_somatic_thresholds[m]
                for m in ["peak_before_to_trough_ratio", "peak_before_to_peak_after_ratio"]
                if m in non_somatic_thresholds
            }
            if len(ratio_thresholds) > 0:
                ratio_condition_labels = threshold_metrics_label_units(
                    sorting_analyzer_or_metrics=combined_metrics,
                    thresholds=ratio_thresholds,
                    pass_label="not_large_ratio",
                    fail_label="large_ratio",
                    operator="and",
                    nan_policy="ignore",
                )
            else:
                ratio_condition_labels = pd.DataFrame(
                    data={"label": np.array(["not_large_ratio"] * len(combined_metrics))}, index=combined_metrics.index
                )

            large_main_peak_thresholds = {
                m: non_somatic_thresholds[m] for m in ["main_peak_to_trough_ratio"] if m in non_somatic_thresholds
            }
            if len(large_main_peak_thresholds) > 0:
                large_main_peak_labels = threshold_metrics_label_units(
                    sorting_analyzer_or_metrics=combined_metrics,
                    thresholds=large_main_peak_thresholds,
                    pass_label="not_large_main_peak",
                    fail_label="large_main_peak",
                    operator="and",
                    nan_policy="ignore",
                )
            else:
                large_main_peak_labels = pd.DataFrame(
                    data={"label": np.array(["not_large_main_peak"] * len(combined_metrics))},
                    index=combined_metrics.index,
                )

            ratio_conditions = ratio_condition_labels["label"] == "large_ratio"
            width_conditions = width_condition_labels["label"] == "narrow_width"
            large_main_peak = large_main_peak_labels["label"] == "large_main_peak"

            # (ratio AND width) OR standalone main_peak_to_trough
            is_non_somatic = (ratio_conditions & width_conditions) | large_main_peak

            if split_non_somatic_good_mua:
                good_mask = unit_labels["label"] == "good"
                mua_mask = unit_labels["label"] == "mua"
                unit_labels.loc[good_mask & is_non_somatic, "label"] = "non_soma_good"
                unit_labels.loc[mua_mask & is_non_somatic, "label"] = "non_soma_mua"
            else:
                noise_mask = unit_labels["label"] == "noise"
                unit_labels.loc[~noise_mask & is_non_somatic, "label"] = "non_soma"
    else:
        # NOISE: waveform failures
        unit_labels = np.full(n_units, fill_value="good", dtype="U10")

        noise_mask = np.zeros(n_units, dtype=bool)
        noise_thresholds = thresholds_dict.get("noise", {})
        for metric_name in noise_thresholds:
            if metric_name not in combined_metrics.columns:
                continue
            values = combined_metrics[metric_name].values
            if metric_name in absolute_value_metrics:
                values = np.abs(values)
            thresh = noise_thresholds[metric_name]
            noise_mask |= np.isnan(values)
            if not is_threshold_disabled(thresh["min"]):
                noise_mask |= values < thresh["min"]
            if not is_threshold_disabled(thresh["max"]):
                noise_mask |= values > thresh["max"]
        unit_labels[noise_mask] = "noise"

        # MUA: spike quality failures
        valid_mask = unit_labels == "good"
        mua_mask = np.zeros(np.sum(valid_mask), dtype=bool)
        mua_thresholds = thresholds_dict.get("mua", {})
        for metric_name in mua_thresholds:
            if metric_name not in combined_metrics.columns:
                continue
            values = combined_metrics[metric_name].values[valid_mask]
            # if metric_name in absolute_value_metrics:
            #     values = np.abs(values)
            num_mua_before = mua_mask.sum()
            thresh = mua_thresholds[metric_name]
            if not is_threshold_disabled(thresh["min"]):
                mua_mask |= ~np.isnan(values) & (values < thresh["min"])
            if not is_threshold_disabled(thresh["max"]):
                mua_mask |= ~np.isnan(values) & (values > thresh["max"])
            num_mua_after = mua_mask.sum()
        valid_indices = np.flatnonzero(valid_mask)
        unit_labels[valid_indices[mua_mask]] = "mua"

        # NON-SOMATIC
        if label_non_somatic:
            non_somatic_thresholds = thresholds_dict.get("non-somatic", {})

            def get_metric(name):
                if name in combined_metrics.columns:
                    return combined_metrics[name].values
                return np.full(n_units, np.nan)

            peak_before_width = get_metric("peak_before_width")
            trough_width = get_metric("trough_width")
            width_thresh_peak = non_somatic_thresholds.get("peak_before_width", {}).get("min", None)
            width_thresh_trough = non_somatic_thresholds.get("trough_width", {}).get("min", None)

            narrow_peak = (
                ~np.isnan(peak_before_width) & (peak_before_width < width_thresh_peak)
                if not is_threshold_disabled(width_thresh_peak)
                else np.zeros(n_units, dtype=bool)
            )
            narrow_trough = (
                ~np.isnan(trough_width) & (trough_width < width_thresh_trough)
                if not is_threshold_disabled(width_thresh_trough)
                else np.zeros(n_units, dtype=bool)
            )
            width_conditions = narrow_peak & narrow_trough

            peak_before_to_trough = get_metric("peak_before_to_trough_ratio")
            peak_before_to_peak_after = get_metric("peak_before_to_peak_after_ratio")
            main_peak_to_trough = get_metric("main_peak_to_trough_ratio")

            ratio_thresh_pbt = non_somatic_thresholds.get("peak_before_to_trough_ratio", {}).get("max", None)
            ratio_thresh_pbpa = non_somatic_thresholds.get("peak_before_to_peak_after_ratio", {}).get("max", None)
            ratio_thresh_mpt = non_somatic_thresholds.get("main_peak_to_trough_ratio", {}).get("max", None)

            large_initial_peak = (
                ~np.isnan(peak_before_to_trough) & (peak_before_to_trough > ratio_thresh_pbt)
                if not is_threshold_disabled(ratio_thresh_pbt)
                else np.zeros(n_units, dtype=bool)
            )
            large_peak_ratio = (
                ~np.isnan(peak_before_to_peak_after) & (peak_before_to_peak_after > ratio_thresh_pbpa)
                if not is_threshold_disabled(ratio_thresh_pbpa)
                else np.zeros(n_units, dtype=bool)
            )
            large_main_peak = (
                ~np.isnan(main_peak_to_trough) & (main_peak_to_trough > ratio_thresh_mpt)
                if not is_threshold_disabled(ratio_thresh_mpt)
                else np.zeros(n_units, dtype=bool)
            )

            # (ratio AND width) OR standalone main_peak_to_trough
            ratio_conditions = large_initial_peak | large_peak_ratio
            is_non_somatic = (ratio_conditions & width_conditions) | large_main_peak

            if split_non_somatic_good_mua:
                unit_labels[(unit_labels == "good") & is_non_somatic] = "non_soma_good"
                unit_labels[(unit_labels == "mua") & is_non_somatic] = "non_soma_mua"
            else:
                unit_labels[(unit_labels != "noise") & is_non_somatic] = "non_soma"
        unit_labels = pd.DataFrame(data={"label": unit_labels}, index=combined_metrics.index)

    return unit_labels


def save_bombcell_results(
    metrics: "pd.DataFrame",
    unit_label: np.ndarray,
    thresholds: dict,
    folder,
    save_narrow: bool = True,
    save_wide: bool = True,
) -> None:
    """
    Save labeling results to CSV files.

    Parameters
    ----------
    metrics : pd.DataFrame
        DataFrame with metrics (index = unit_ids).
    unit_label : np.ndarray
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

    unit_ids = metrics.index.values

    # Wide format: one row per unit
    if save_wide:
        wide_df = metrics.copy()
        wide_df.insert(0, "label", unit_label)
        wide_df.to_csv(folder / "labeling_results_wide.csv")

    # Flatten thresholds for saving
    flat_thresholds = {}
    for category, metric_dict in thresholds.items():
        for metric_name, thresh in metric_dict.items():
            flat_thresholds[metric_name] = thresh

    # Narrow format: one row per unit-metric combination
    if save_narrow:
        rows = []
        for i, unit_id in enumerate(unit_ids):
            label = unit_label[i]
            for metric_name in metrics.columns:
                if metric_name not in flat_thresholds:
                    continue
                value = metrics.loc[unit_id, metric_name]
                thresh = flat_thresholds[metric_name]
                thresh_min = thresh.get("min", None)
                thresh_max = thresh.get("max", None)

                # Determine pass/fail
                passed = True
                if np.isnan(value):
                    passed = False
                elif not is_threshold_disabled(thresh_min) and value < thresh_min:
                    passed = False
                elif not is_threshold_disabled(thresh_max) and value > thresh_max:
                    passed = False

                rows.append(
                    {
                        "unit_id": unit_id,
                        "label": label,
                        "metric_name": metric_name,
                        "value": value,
                        "threshold_min": thresh_min,
                        "threshold_max": thresh_max,
                        "passed": passed,
                    }
                )

        narrow_df = pd.DataFrame(rows)
        narrow_df.to_csv(folder / "labeling_results_narrow.csv", index=False)
