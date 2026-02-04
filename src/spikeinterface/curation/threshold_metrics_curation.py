import json
from pathlib import Path

import numpy as np

from spikeinterface.core.analyzer_extension_core import SortingAnalyzer

from .curation_tools import is_threshold_disabled


def threshold_metrics_label_units(
    sorting_analyzer_or_metrics: "SortingAnalyzer | pd.DataFrame",
    thresholds: dict | str | Path,
    pass_label: str = "good",
    fail_label: str = "noise",
):
    """Label units based on metrics and thresholds.

    Parameters
    ----------
    sorting_analyzer_or_metrics : SortingAnalyzer | pd.DataFrame
        The SortingAnalyzer object containing the some metrics extensions (e.g., quality metrics) or a DataFrame
        containing unit metrics with unit IDs as index.
    thresholds : dict | str | Path
        A dictionary or JSON file path where keys are metric names and values are threshold values for labeling units.
        Each key should correspond to a quality metric present in the analyzer's quality metrics DataFrame. Values
        should contain at least "min" and/or "max" keys to specify threshold ranges.
    pass_label : str, default: "good"
        The label to assign to units that pass all thresholds.
    fail_label : str, default: "noise"
        The label to assign to units that fail any threshold.

    Returns
    -------
    labels : pd.DataFrame
        A DataFrame with unit IDs as index and a column 'label' containing the assigned labels ('noise' or 'good').
    """
    import pandas as pd

    if not isinstance(sorting_analyzer_or_metrics, (SortingAnalyzer, pd.DataFrame)):
        raise ValueError("Only SortingAnalyzer or pd.DataFrame are supported for sorting_analyzer_or_metrics.")

    if isinstance(sorting_analyzer_or_metrics, SortingAnalyzer):
        metrics = sorting_analyzer_or_metrics.get_metrics_extension_data()
    else:
        metrics = sorting_analyzer_or_metrics

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

    # Initialize an empty DataFrame to store labels
    labels = pd.DataFrame(index=metrics.index, dtype=str)
    labels["label"] = fail_label

    # Apply thresholds to label units
    pass_mask = np.ones(len(metrics), dtype=bool)
    for metric_name, threshold in thresholds_dict.items():
        min_value = threshold.get("min", None)
        max_value = threshold.get("max", None)
        if not is_threshold_disabled(min_value):
            pass_mask &= metrics[metric_name] >= min_value
        if not is_threshold_disabled(max_value):
            pass_mask &= metrics[metric_name] <= max_value

    labels.loc[pass_mask, "label"] = pass_label
    return labels
