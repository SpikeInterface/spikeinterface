import json
from pathlib import Path

import numpy as np

from spikeinterface.core.analyzer_extension_core import SortingAnalyzer

from .curation_tools import is_threshold_disabled


def threshold_metrics_label_units(
    metrics: "pd.DataFrame",
    thresholds: dict | str | Path,
    pass_label: str = "good",
    fail_label: str = "noise",
    operator: str = "and",
    nan_policy: str = "fail",
    column_name: str = "label",
):
    """Label units based on metrics and thresholds.

    Parameters
    ----------
    metrics : pd.DataFrame
        A DataFrame containing unit metrics with unit IDs as index.
    thresholds : dict | str | Path
        A dictionary or JSON file path where keys are metric names and values are threshold values for labeling units.
        Each key should correspond to a quality metric present in the analyzer's quality metrics DataFrame. Values
        should contain at least "greater" and/or "less" keys to specify threshold ranges. Thresholds are inclusive, i.e.
        "greater" is >= and "less" is <=. Optionally, an "abs": True entry can be included to indicate that the metric
        should be treated as an absolute value when applying thresholds.
    pass_label : str, default: "good"
        The label to assign to units that pass all thresholds.
    fail_label : str, default: "noise"
        The label to assign to units that fail any threshold.
    operator : "and" | "or", default: "and"
        The logical operator to combine multiple metric thresholds. "and" means a unit must pass all thresholds to be
        labeled as pass_label, while "or" means a unit must pass at least one threshold to be labeled as pass_label.
    nan_policy : "fail" | "pass" | "ignore", default: "fail"
        Policy for handling NaN values in metrics. If "fail", units with NaN values in any metric will be labeled as
        fail_label. If "pass", units with NaN values in one metric will be labeled as pass_label.
        If "ignore", NaN values will be ignored. Note that the "ignore" behavior will depend on the operator used.
        If "and", NaNs will be treated as passing, since the initial mask is all true;
        if "or", NaNs will be treated as failing, since the initial mask is all false.
    column_name : str, default: "label"
        The name of the column in the output DataFrame that will contain the assigned labels.

    Returns
    -------
    labels : pd.DataFrame
        A DataFrame with unit IDs as index and a column `column_name` containing the assigned labels (`fail_label` or `pass_label`)
    """
    import pandas as pd

    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Only pd.DataFrame is supported for metrics.")

    # Load thresholds from file if a path is provided
    if isinstance(thresholds, (str, Path)):
        with open(thresholds, "r") as f:
            thresholds_dict = json.load(f)
    elif isinstance(thresholds, dict):
        thresholds_dict = thresholds
    else:
        raise ValueError("Thresholds must be a dictionary or a path to a JSON file containing the thresholds.")

    # Check that all specified metrics are present in the quality metrics DataFrame
    missing_metrics = []
    for metric in thresholds_dict.keys():
        if metric not in metrics.columns:
            missing_metrics.append(metric)
    if len(missing_metrics) > 0:
        raise ValueError(
            f"Metric(s) {missing_metrics} specified in thresholds are not present in the quality metrics DataFrame. "
            f"Available metrics are: {metrics.columns.tolist()}"
        )

    # Check that threshold dictionaries contain only valid keys
    valid_keys = {"greater", "less", "abs"}
    for metric_name, threshold in thresholds_dict.items():
        if not set(threshold).issubset(valid_keys):
            raise ValueError(
                f"Threshold for metric '{metric_name}' contains invalid keys {set(threshold) - valid_keys}."
            )

    if operator not in ("and", "or"):
        raise ValueError("operator must be 'and' or 'or'")

    if nan_policy not in ("fail", "pass", "ignore"):
        raise ValueError("nan_policy must be 'fail', 'pass', or 'ignore'")

    labels = pd.DataFrame(index=metrics.index, dtype=str)
    labels[column_name] = fail_label

    # Key change: init depends on operator
    pass_mask = np.ones(len(metrics), dtype=bool) if operator == "and" else np.zeros(len(metrics), dtype=bool)
    any_threshold_applied = False

    for metric_name, threshold in thresholds_dict.items():
        min_value = threshold.get("greater", None)
        max_value = threshold.get("less", None)
        abs_value = threshold.get("abs", False)

        # If both disabled, ignore this metric
        if is_threshold_disabled(min_value) and is_threshold_disabled(max_value):
            continue

        values = metrics[metric_name].to_numpy()
        if abs_value:
            values = np.abs(values)
        is_nan = np.isnan(values)

        metric_ok = np.ones(len(values), dtype=bool)
        if not is_threshold_disabled(min_value):
            metric_ok &= values >= min_value
        if not is_threshold_disabled(max_value):
            metric_ok &= values <= max_value

        # Handle NaNs
        if nan_policy == "fail":
            metric_ok &= ~is_nan
            valid_mask = slice(None)
        elif nan_policy == "pass":
            metric_ok |= is_nan
            valid_mask = slice(None)
        elif nan_policy == "ignore":
            valid_mask = ~is_nan

        any_threshold_applied = True

        if operator == "and":
            pass_mask[valid_mask] &= metric_ok[valid_mask]
        elif operator == "or":
            pass_mask[valid_mask] |= metric_ok[valid_mask]

    if not any_threshold_applied:
        pass_mask[:] = True

    labels.loc[pass_mask, column_name] = pass_label
    return labels
