import pytest
import json

import numpy as np

from spikeinterface.curation import threshold_metrics_label_units


def test_threshold_metrics_label_units_with_dataframe():
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "snr": [6.0, 4.0, 5.0],
            "firing_rate": [0.5, 0.2, 25.0],
        },
        index=[0, 1, 2],
    )
    thresholds = {
        "snr": {"greater": 5.0},
        "firing_rate": {"greater": 0.1, "less": 20.0},
    }

    labels = threshold_metrics_label_units(metrics, thresholds)

    assert "label" in labels.columns
    assert labels.shape[0] == len(metrics.index)
    assert labels["label"].to_dict() == {0: "good", 1: "noise", 2: "noise"}


def test_threshold_metrics_label_units_with_file(tmp_path):
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "snr": [6.0, 4.0],
            "firing_rate": [0.5, 0.05],
        },
        index=[0, 1],
    )
    thresholds = {
        "snr": {"greater": 5.0},
        "firing_rate": {"greater": 0.1},
    }

    thresholds_file = tmp_path / "thresholds.json"
    with open(thresholds_file, "w") as f:
        json.dump(thresholds, f)

    labels = threshold_metrics_label_units(metrics, thresholds_file)

    assert labels["label"].to_dict() == {0: "good", 1: "noise"}


def test_threshold_metrics_label_external_labels():
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "snr": [6.0, 4.0],
            "firing_rate": [0.5, 0.05],
        },
        index=[0, 1],
    )
    thresholds = {
        "snr": {"greater": 5.0},
        "firing_rate": {"greater": 0.1},
    }

    labels = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        pass_label="accepted",
        fail_label="rejected",
    )
    assert set(labels["label"]).issubset({"accepted", "rejected"})


def test_threshold_metrics_label_units_operator_or_with_dataframe():
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "m1": [1.0, 1.0, -1.0, -1.0],
            "m2": [1.0, -1.0, 1.0, -1.0],
        },
        index=[0, 1, 2, 3],
    )
    thresholds = {"m1": {"greater": 0.0}, "m2": {"greater": 0.0}}

    labels_and = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        operator="and",
    )
    assert labels_and.index.equals(metrics.index)
    assert labels_and["label"].to_dict() == {0: "good", 1: "noise", 2: "noise", 3: "noise"}

    labels_or = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        operator="or",
    )
    assert labels_or.index.equals(metrics.index)
    assert labels_or["label"].to_dict() == {0: "good", 1: "good", 2: "good", 3: "noise"}


def test_threshold_metrics_label_units_nan_policy_fail_vs_ignore_and():
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "m1": [np.nan, 1.0, np.nan],
            "m2": [1.0, -1.0, -1.0],
        },
        index=[10, 11, 12],
    )
    thresholds = {"m1": {"greater": 0.0}, "m2": {"greater": 0.0}}

    labels_fail = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        operator="and",
        nan_policy="fail",
    )
    assert labels_fail["label"].to_dict() == {10: "noise", 11: "noise", 12: "noise"}

    labels_ignore = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        operator="and",
        nan_policy="ignore",
    )
    # unit 10: m1 ignored (NaN), m2 passes -> good
    # unit 11: m2 fails -> noise
    # unit 12: m1 ignored but m2 fails -> noise
    assert labels_ignore["label"].to_dict() == {10: "good", 11: "noise", 12: "noise"}


def test_threshold_metrics_label_units_nan_policy_ignore_with_or():
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "m1": [np.nan, -1.0],
            "m2": [-1.0, -1.0],
        },
        index=[20, 21],
    )
    thresholds = {"m1": {"greater": 0.0}, "m2": {"greater": 0.0}}

    labels_ignore_or = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        operator="or",
        nan_policy="ignore",
    )
    # unit 20: m1 is NaN and ignored; m2 fails => noise
    # unit 21: both metrics fail => noise
    assert labels_ignore_or["label"].to_dict() == {20: "noise", 21: "noise"}


def test_threshold_metrics_label_units_nan_policy_pass_and_or():
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "m1": [np.nan, np.nan, 1.0, -1.0],
            "m2": [1.0, -1.0, np.nan, np.nan],
        },
        index=[30, 31, 32, 33],
    )
    thresholds = {"m1": {"greater": 0.0}, "m2": {"greater": 0.0}}

    labels_and = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        operator="and",
        nan_policy="pass",
    )
    # unit 30: m1 NaN (pass), m2 pass => good
    # unit 31: m1 NaN (pass), m2 fail => noise
    # unit 32: m1 pass, m2 NaN (pass) => good
    # unit 33: m1 fail, m2 NaN (pass) => noise
    assert labels_and["label"].to_dict() == {30: "good", 31: "noise", 32: "good", 33: "noise"}

    labels_or = threshold_metrics_label_units(
        metrics,
        thresholds=thresholds,
        operator="or",
        nan_policy="pass",
    )
    # any NaN counts as pass => good unless all metrics fail without NaN
    assert labels_or["label"].to_dict() == {30: "good", 31: "good", 32: "good", 33: "good"}


def test_threshold_metrics_label_units_invalid_operator_raises():
    import pandas as pd

    metrics = pd.DataFrame({"m1": [1.0]}, index=[0])
    thresholds = {"m1": {"greater": 0.0}}
    with pytest.raises(ValueError, match="operator must be 'and' or 'or'"):
        threshold_metrics_label_units(metrics, thresholds, operator="xor")


def test_threshold_metrics_label_units_invalid_nan_policy_raises():
    import pandas as pd

    metrics = pd.DataFrame({"m1": [1.0]}, index=[0])
    thresholds = {"m1": {"greater": 0.0}}
    with pytest.raises(ValueError, match="nan_policy must be"):
        threshold_metrics_label_units(metrics, thresholds, nan_policy="omit")


def test_threshold_metrics_label_units_missing_metric_raises():
    import pandas as pd

    metrics = pd.DataFrame({"m1": [1.0]}, index=[0])
    thresholds = {"does_not_exist": {"greater": 0.0}}
    with pytest.raises(ValueError, match="specified in thresholds are not present"):
        threshold_metrics_label_units(metrics, thresholds)


def test_threshold_metrics_label_units_invalid_threshold_keys_raises():
    import pandas as pd

    metrics = pd.DataFrame({"m1": [1.0]}, index=[0])
    thresholds = {"m1": {"greater": 0.0, "invalid_key": 1.0}}
    with pytest.raises(ValueError, match="contains invalid keys"):
        threshold_metrics_label_units(metrics, thresholds)
