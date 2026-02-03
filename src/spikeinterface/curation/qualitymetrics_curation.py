import json
from pathlib import Path

import numpy as np

from spikeinterface.core.analyzer_extension_core import SortingAnalyzer

from .curation_tools import _is_threshold_disabled


def qualitymetrics_label_units(
    analyzer: SortingAnalyzer,
    thresholds: dict | str | Path,
):
    """Label units based on quality metrics and thresholds.

    Parameters
    ----------
    analyzer : SortingAnalyzer
        The SortingAnalyzer object containing the quality metrics.
    thresholds : dict | str | Path
        A dictionary or JSON file path where keys are metric names and values are threshold values for labeling units.
        Each key should correspond to a quality metric present in the analyzer's quality metrics DataFrame. Values
        should contain at least "min" and/or "max" keys to specify threshold ranges.
        Units that do not meet the threshold for a given metric will be labeled as 'noise', while those that do will
        be labeled as 'good'.

    Returns
    -------
    labels : pd.DataFrame
        A DataFrame with unit IDs as index and a column 'label' containing the assigned labels ('noise' or 'good').
    """
    import pandas as pd

    # Get the quality metrics from the analyzer
    assert analyzer.has_extension("quality_metrics"), (
        "The provided analyzer does not have quality metrics computed. "
        "Please compute quality metrics before labeling units."
    )
    qm = analyzer.get_extension("quality_metrics").get_data()

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
        if metric not in qm.columns:
            missing_metrics.append(metric)
    if len(missing_metrics) > 0:
        raise ValueError(
            f"Metric(s) {missing_metrics} specified in thresholds are not present in the quality metrics DataFrame. "
            f"Available metrics are: {qm.columns.tolist()}"
        )

    # Initialize an empty DataFrame to store labels
    labels = pd.DataFrame(index=qm.index, dtype=str)
    labels["label"] = "noise"  # Default label is 'noise'

    # Apply thresholds to label units
    good_mask = np.ones(len(qm), dtype=bool)

    for metric_name, threshold in thresholds_dict.items():
        min_value = threshold.get("min", None)
        max_value = threshold.get("max", None)
        if not _is_threshold_disabled(min_value):
            good_mask &= qm[metric_name] >= min_value
        if not _is_threshold_disabled(max_value):
            good_mask &= qm[metric_name] <= max_value

    labels.loc[good_mask, "label"] = "good"

    return labels
