import pytest
import json

import numpy as np
import pandas as pd

from spikeinterface.curation.tests.common import sorting_analyzer_for_curation
from spikeinterface.curation import threshold_metrics_label_units


@pytest.fixture
def sorting_analyzer_with_metrics(sorting_analyzer_for_curation):
    """A sorting analyzer with computed quality metrics."""

    sorting_analyzer = sorting_analyzer_for_curation
    sorting_analyzer.compute("quality_metrics")
    return sorting_analyzer


def test_threshold_metrics_label_units(sorting_analyzer_with_metrics):
    """Test the `threshold_metrics_label_units` function."""
    sorting_analyzer = sorting_analyzer_with_metrics

    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1, "max": 20.0},
    }

    labels = threshold_metrics_label_units(
        sorting_analyzer,
        thresholds,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    qm = sorting_analyzer.get_extension("quality_metrics").get_data()
    for unit_id in sorting_analyzer.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if (
            snr >= thresholds["snr"]["min"]
            and thresholds["firing_rate"]["min"] <= firing_rate <= thresholds["firing_rate"]["max"]
        ):
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"


def test_threshold_metrics_label_units_with_file(sorting_analyzer_with_metrics, tmp_path):
    """Test the `threshold_metrics_label_units` function with thresholds from a JSON file."""
    sorting_analyzer = sorting_analyzer_with_metrics

    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1},
    }

    thresholds_file = tmp_path / "thresholds.json"
    with open(thresholds_file, "w") as f:
        json.dump(thresholds, f)

    labels = threshold_metrics_label_units(
        sorting_analyzer,
        thresholds_file,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    qm = sorting_analyzer.get_extension("quality_metrics").get_data()
    for unit_id in sorting_analyzer.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if snr >= thresholds["snr"]["min"] and firing_rate >= thresholds["firing_rate"]["min"]:
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"


def test_threshold_metrics_label_units_with_external_metrics(sorting_analyzer_with_metrics):
    """Test the `threshold_metrics_label_units` function with external metrics DataFrame."""
    sorting_analyzer = sorting_analyzer_with_metrics
    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1, "max": 20.0},
    }

    qm = sorting_analyzer.get_extension("quality_metrics").get_data()

    labels = threshold_metrics_label_units(
        sorting_analyzer_or_metrics=qm,
        thresholds=thresholds,
    )

    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)

    # Check that units with snr < 5.0 or firing_rate < 0.1 are labeled as 'noise'
    for unit_id in sorting_analyzer.sorting.unit_ids:
        snr = qm.loc[unit_id, "snr"]
        firing_rate = qm.loc[unit_id, "firing_rate"]
        if (
            snr >= thresholds["snr"]["min"]
            and thresholds["firing_rate"]["min"] <= firing_rate <= thresholds["firing_rate"]["max"]
        ):
            assert labels.loc[unit_id, "label"] == "good"
        else:
            assert labels.loc[unit_id, "label"] == "noise"


def test_threshold_metrics_label_external_labels(sorting_analyzer_with_metrics):
    """Test the `threshold_metrics_label_units` function with custom pass/fail labels."""
    sorting_analyzer = sorting_analyzer_with_metrics
    thresholds = {
        "snr": {"min": 5.0},
        "firing_rate": {"min": 0.1, "max": 20.0},
    }

    labels = threshold_metrics_label_units(
        sorting_analyzer,
        thresholds=thresholds,
        pass_label="accepted",
        fail_label="rejected",
    )
    assert "label" in labels.columns
    assert labels.shape[0] == len(sorting_analyzer.sorting.unit_ids)
    assert set(labels["label"]).issubset({"accepted", "rejected"})


def test_threshold_metrics_label_units_operator_or_with_dataframe():
    metrics = pd.DataFrame(
        {
            "m1": [1.0, 1.0, -1.0, -1.0],
            "m2": [1.0, -1.0, 1.0, -1.0],
        },
        index=[0, 1, 2, 3],
    )
    thresholds = {"m1": {"min": 0.0}, "m2": {"min": 0.0}}

    labels_and = threshold_metrics_label_units(
        sorting_analyzer_or_metrics=metrics,
        thresholds=thresholds,
        operator="and",
    )
    assert labels_and.index.equals(metrics.index)
    assert labels_and["label"].to_dict() == {0: "good", 1: "noise", 2: "noise", 3: "noise"}

    labels_or = threshold_metrics_label_units(
        sorting_analyzer_or_metrics=metrics,
        thresholds=thresholds,
        operator="or",
    )
    assert labels_or.index.equals(metrics.index)
    assert labels_or["label"].to_dict() == {0: "good", 1: "good", 2: "good", 3: "noise"}


def test_threshold_metrics_label_units_nan_policy_fail_vs_ignore_and():
    metrics = pd.DataFrame(
        {
            "m1": [np.nan, 1.0, np.nan],
            "m2": [1.0, -1.0, -1.0],
        },
        index=[10, 11, 12],
    )
    thresholds = {"m1": {"min": 0.0}, "m2": {"min": 0.0}}

    labels_fail = threshold_metrics_label_units(
        sorting_analyzer_or_metrics=metrics,
        thresholds=thresholds,
        operator="and",
        nan_policy="fail",
    )
    assert labels_fail["label"].to_dict() == {10: "noise", 11: "noise", 12: "noise"}

    labels_ignore = threshold_metrics_label_units(
        sorting_analyzer_or_metrics=metrics,
        thresholds=thresholds,
        operator="and",
        nan_policy="ignore",
    )
    # unit 10: m1 ignored (NaN), m2 passes -> good
    # unit 11: m2 fails -> noise
    # unit 12: m1 ignored but m2 fails -> noise
    assert labels_ignore["label"].to_dict() == {10: "good", 11: "noise", 12: "noise"}


def test_threshold_metrics_label_units_nan_policy_ignore_with_or():
    metrics = pd.DataFrame(
        {
            "m1": [np.nan, -1.0],
            "m2": [-1.0, -1.0],
        },
        index=[20, 21],
    )
    thresholds = {"m1": {"min": 0.0}, "m2": {"min": 0.0}}

    labels_ignore_or = threshold_metrics_label_units(
        sorting_analyzer_or_metrics=metrics,
        thresholds=thresholds,
        operator="or",
        nan_policy="ignore",
    )
    # unit 20: m1 is NaN and ignored => passes that metric => good under "or"
    # unit 21: both metrics fail => noise
    assert labels_ignore_or["label"].to_dict() == {20: "good", 21: "noise"}


def test_threshold_metrics_label_units_invalid_operator_raises():
    metrics = pd.DataFrame({"m1": [1.0]}, index=[0])
    thresholds = {"m1": {"min": 0.0}}
    with pytest.raises(ValueError, match="operator must be 'and' or 'or'"):
        threshold_metrics_label_units(metrics, thresholds, operator="xor")


def test_threshold_metrics_label_units_invalid_nan_policy_raises():
    metrics = pd.DataFrame({"m1": [1.0]}, index=[0])
    thresholds = {"m1": {"min": 0.0}}
    with pytest.raises(ValueError, match="nan_policy must be 'fail' or 'ignore'"):
        threshold_metrics_label_units(metrics, thresholds, nan_policy="omit")


def test_threshold_metrics_label_units_missing_metric_raises():
    metrics = pd.DataFrame({"m1": [1.0]}, index=[0])
    thresholds = {"does_not_exist": {"min": 0.0}}
    with pytest.raises(ValueError, match="specified in thresholds are not present"):
        threshold_metrics_label_units(metrics, thresholds)
